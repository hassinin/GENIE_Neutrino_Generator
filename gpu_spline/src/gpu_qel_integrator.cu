#include "gpu_spline/gpu_qel_integrator.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <limits>

namespace gpu_spline {
namespace {
// Throws GpuDeviceError (and honours fault injection) like GPU_CHECK.
void check(gpuError_t e) { gpuAssert(e,__FILE__,__LINE__); }
template<class T> struct Buffer {
    T* data=nullptr;
    explicit Buffer(size_t n) { check(gpuMalloc(reinterpret_cast<void**>(&data),n*sizeof(T))); }
    ~Buffer() { if(data) (void)gpuFree(data); }
    Buffer(const Buffer&)=delete;
    Buffer& operator=(const Buffer&)=delete;
    void upload(const T* p,size_t n) { check(gpuMemcpy(data,p,n*sizeof(T),gpuMemcpyHostToDevice)); }
};
void gauss_legendre(int n,std::vector<double>& x,std::vector<double>& w) {
    x.resize(n); w.resize(n);
    for(int i=0;i<(n+1)/2;++i) {
        double z=std::cos(3.14159265358979323846*(i+.75)/(n+.5)), derivative=0;
        for(int it=0;it<100;++it) {
            double a=1,b=0;
            for(int j=1;j<=n;++j) { double c=b; b=a; a=((2*j-1)*z*b-(j-1)*c)/j; }
            derivative=n*(z*a-b)/(z*z-1);
            double delta=a/derivative; z-=delta;
            if(std::abs(delta)<4e-16) break;
            if(it==99) throw std::runtime_error("Gauss-Legendre nodes failed to converge");
        }
        x[i]=-z; x[n-i-1]=z;
        w[i]=w[n-i-1]=2/((1-z*z)*derivative*derivative);
    }
}
// In the probe-projected azimuthal basis Q2=A0+A1*c-D*sqrt(1-c*c)*cos(phi).
// Split c wherever either Q2 bound touches a phi endpoint, and where the
// de Forest lower bound crosses Q2min. This resolves narrow angular caps and
// keeps square-root cusps out of the interior of a quadrature interval.
struct Domain { int intervals=0; double edge[9]{}; };
Domain domain(const QelParameters& p,const QelSample& s) {
    Domain out;
    if(!s.valid) return out;
    const double ka=p.probe.x*s.axis.x+p.probe.y*s.axis.y+p.probe.z*s.axis.z;
    const double kt=p.probe.x*s.transverse.x+p.probe.y*s.transverse.y+p.probe.z*s.transverse.z;
    const double a0=2*s.gamma*s.lepton_com*(p.probe.t-ka*s.beta)-p.lepton_mass*p.lepton_mass;
    const double a1=2*s.gamma*s.momentum*(p.probe.t*s.beta-ka);
    const double d=2*s.momentum*std::max(0.,kt);
    const double eps=s.on_shell.t-s.initial.t;
    const double l0=eps*eps-2*eps*(p.probe.t-s.gamma*s.lepton_com);
    const double l1=2*eps*s.gamma*s.beta*s.momentum;
    double cmin=p.nieves?s.cos_min:-1.;
    std::vector<double> edges{cmin,s.cos_max};
    auto add=[&](double c) { if(c>cmin && c<s.cos_max) edges.push_back(c); };
    auto roots=[&](double lower0,double lower1) {
        double b0=a0-lower0,b1=a1-lower1,den=b1*b1+d*d;
        if(den==0) return;
        double rad=den-b0*b0;
        if(rad<0) return;
        double term=d*std::sqrt(rad);
        add((-b0*b1-term)/den); add((-b0*b1+term)/den);
    };
    roots(s.q2min,0); roots(s.q2max,0);
    if(!p.nieves) { roots(l0,l1); if(l1!=0) add((s.q2min-l0)/l1); }
    std::sort(edges.begin(),edges.end());
    edges.erase(std::unique(edges.begin(),edges.end()),edges.end());
    out.intervals=int(edges.size())-1;
    std::copy(edges.begin(),edges.end(),out.edge);
    return out;
}
__global__ void integrate(QelParameters p,const QelSample* samples,const Domain* domains,const int* active,
                          const double* nodes,const double* weights,int order,double* output) {
    int id=blockIdx.x;
    if(!active[id]) return;
    const QelSample& s=samples[id];
    const Domain& part=domains[id];
    double sum=0;
    for(int j=threadIdx.x;j<order*order*part.intervals;j+=blockDim.x) {
        int segment=j/(order*order),a=(j/order)%order,b=j%order;
        double half=(part.edge[segment+1]-part.edge[segment])/2;
        double mid=(part.edge[segment+1]+part.edge[segment])/2;
        const double pi2=1.57079632679489661923;
        // A sine map removes the endpoint square-root cusp introduced by
        // the analytic phi limits and gives smooth tensor quadrature.
        double c=mid+half*sin(pi2*nodes[a]),jac=half*pi2*cos(pi2*nodes[a]),lo=0,hi=0;
        if(qel_phi_limits(p,s,c,lo,hi)) {
            double phi=(lo+hi)/2+(hi-lo)*nodes[b]/2;
            sum+=weights[a]*weights[b]*jac*(hi-lo)*qel_folded_integrand(p,s,c,phi);
        }
    }
    __shared__ double total[256];
    total[threadIdx.x]=sum; __syncthreads();
    for(int step=blockDim.x/2;step;step/=2) {
        if(threadIdx.x<step) total[threadIdx.x]+=total[threadIdx.x+step];
        __syncthreads();
    }
    if(threadIdx.x==0) output[id]=total[0];
}
// A rectangle in the transformed (polar, azimuthal) coordinates of one
// analytic angular interval. Subdivision preserves that interval's sine map.
struct Rectangle { int sample; double c0,c1,u0,u1,v0,v1; };
struct Estimate { double value,error,polar_error,azimuth_error; };
struct Leaf { Rectangle rect; Estimate estimate; bool verified=false; };
GPU_DEVICE double rectangle_value(const QelParameters& p,const QelSample& s,
                                  const Rectangle& r,double x,double y) {
    const double pi2=1.57079632679489661923;
    double u=(r.u0+r.u1)/2+(r.u1-r.u0)*x/2;
    double v=(r.v0+r.v1)/2+(r.v1-r.v0)*y/2;
    double half=(r.c1-r.c0)/2,c=(r.c0+r.c1)/2+half*sin(pi2*u);
    double lo=0,hi=0;
    if(!qel_phi_limits(p,s,c,lo,hi)) return 0.;
    double jac=half*pi2*cos(pi2*u)*(r.u1-r.u0)*(r.v1-r.v0)*(hi-lo)/2;
    return jac*qel_folded_integrand(p,s,c,lo+(hi-lo)*v);
}
// 8x8, 16x16, 8x16, 16x8 rules give a local error and split direction.
// A separate 32x32 rule verifies every final leaf before acceptance.
__global__ void rectangles(QelParameters p,const QelSample* samples,const Rectangle* rects,
                           const double* nodes,const double* weights,bool verify,Estimate* output) {
    int id=blockIdx.x; const Rectangle& r=rects[id];
    double sum[4]{};
    int count=verify?1024:576;
    for(int j=threadIdx.x;j<count;j+=blockDim.x) {
        int rule=0,k=j,ny=32,ox=24,oy=24;
        if(!verify) {
            if(j<64) { ny=8;ox=oy=0; }
            else if(j<320) { rule=1;k=j-64;ny=16;ox=oy=8; }
            else if(j<448) { rule=2;k=j-320;ny=16;ox=0;oy=8; }
            else { rule=3;k=j-448;ny=8;ox=8;oy=0; }
        }
        int a=k/ny,b=k%ny;
        sum[rule]+=weights[ox+a]*weights[oy+b]*rectangle_value(p,samples[r.sample],r,nodes[ox+a],nodes[oy+b]);
    }
    __shared__ double total[4][128];
    for(int k=0;k<4;++k) total[k][threadIdx.x]=sum[k];
    __syncthreads();
    for(int step=blockDim.x/2;step;step/=2) {
        if(threadIdx.x<step) for(int k=0;k<4;++k) total[k][threadIdx.x]+=total[k][threadIdx.x+step];
        __syncthreads();
    }
    if(threadIdx.x==0) {
        Estimate e{};
        if(verify) e.value=total[0][0];
        else {
            e.value=total[1][0];
            e.polar_error=fabs(e.value-total[2][0]);
            e.azimuth_error=fabs(e.value-total[3][0]);
            e.error=fmax(fabs(e.value-total[0][0]),e.polar_error+e.azimuth_error);
            e.error=fmax(e.error,32*2.2204460492503131e-16*fabs(e.value));
        }
        output[id]=e;
    }
}
__global__ void evaluate(QelParameters p,const QelSample* samples,const double* c,
                         const double* phi,int n,double* output) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) output[i]=qel_integrand(p,samples[i],c[i],phi[i]);
}
}
bool IntegrateQelGpu(const QelParameters& p,const std::vector<QelSample>& samples,
                     double tolerance,unsigned budget,int device,
                     std::vector<QelIntegralResult>& result,std::string& error,bool* device_error) {
    const unsigned cost_factor=p.nieves?2:1;
    result.assign(samples.size(),{}); error.clear(); if(device_error) *device_error=false;
    if(samples.empty()) return true;
    if(!(tolerance>0 && tolerance<1) || samples.size()>2147483647u) { error="Invalid QEL integration request"; return false; }
    try {
        check(gpuSetDevice(device));
        const size_t n=samples.size();
        Buffer<QelSample> ds(n); ds.upload(samples.data(),n);
        std::vector<Domain> parts; parts.reserve(n);
        for(const auto& sample:samples) parts.push_back(domain(p,sample));
        Buffer<Domain> dd(n); dd.upload(parts.data(),n);
        Buffer<int> da(n); Buffer<double> dout(n),dx(256),dw(256);
        check(gpuMemset(dout.data,0,n*sizeof(double)));
        std::vector<int> active(n), stable(n,0);
        std::vector<double> value(n),previous(n,0),x,w;
        for(size_t i=0;i<n;++i) { active[i]=samples[i].valid; result[i].converged=!active[i]; }
        for(int order=16;order<=256;order*=2) {
            for(size_t i=0;i<n;++i) if(active[i] &&
                result[i].evaluations+unsigned(order*order*parts[i].intervals)*cost_factor>budget) active[i]=0;
            if(std::none_of(active.begin(),active.end(),[](int a){return a!=0;})) break;
            gauss_legendre(order,x,w); dx.upload(x.data(),order); dw.upload(w.data(),order); da.upload(active.data(),n);
            integrate<<<n,256>>>(p,ds.data,dd.data,da.data,dx.data,dw.data,order,dout.data);
            check(gpuGetLastError()); check(gpuDeviceSynchronize());
            check(gpuMemcpy(value.data(),dout.data,n*sizeof(double),gpuMemcpyDeviceToHost));
            for(size_t i=0;i<n;++i) if(active[i]) {
                auto& r=result[i];
                r.value=value[i]; r.error=std::abs(value[i]-previous[i]); r.evaluations+=order*order*parts[i].intervals*cost_factor;
                // Successive-order differences can underestimate the actual error
                // near Q2-cut cusps. Use a tenfold margin on the requested tolerance.
                const double threshold=std::max(1e-24,0.1*tolerance*std::abs(value[i]));
                if(order>16 && std::isfinite(value[i]) && value[i]>0 && r.error<=threshold) ++stable[i];
                else stable[i]=0;
                // Require two successive refinements. Numerically zero integrals
                // with a potentially nonempty domain are checked by the CPU.
                r.converged=stable[i]>=2;
                if(r.converged) active[i]=0;
                previous[i]=value[i];
            }
        }
        return true;
    } catch(const GpuDeviceError& e) { error=e.what(); if(device_error) *device_error=true; return false; }
    catch(const std::exception& e) { error=e.what(); (void)gpuGetLastError(); return false; }
}
bool RefineQelGpu(const QelParameters& p,const std::vector<QelSample>& samples,
                  double tolerance,unsigned budget,int device,
                  std::vector<QelIntegralResult>& result,std::string& error,bool* device_error) {
    error.clear(); if(device_error) *device_error=false;
    if(samples.size()!=result.size() || samples.size()>2147483647u || !(tolerance>0 && tolerance<1)) {
        error="Invalid adaptive QEL integration request"; return false;
    }
    if(!budget || samples.empty()) return true;
    const unsigned cost_factor=p.nieves?2:1;
    try {
        // Work on a copy so any CUDA failure preserves all first-stage results.
        auto refined=result;
        std::vector<std::vector<Leaf>> leaves(samples.size());
        std::vector<unsigned> used(samples.size(),0);
        std::vector<int> active(samples.size(),0);
        std::vector<Rectangle> tasks;
        for(size_t i=0;i<samples.size();++i) if(samples[i].valid && !result[i].converged) {
            auto part=domain(p,samples[i]);
            unsigned cost=576*part.intervals*cost_factor;
            if(!cost || cost>budget || budget>std::numeric_limits<unsigned>::max()-result[i].evaluations) continue;
            active[i]=1;used[i]=cost;
            for(int j=0;j<part.intervals;++j) {
                Rectangle r{int(i),part.edge[j],part.edge[j+1],-1,1,0,1};
                leaves[i].push_back({r,{},false});tasks.push_back(r);
            }
        }
        if(tasks.empty()) return true;
        check(gpuSetDevice(device));
        Buffer<QelSample> ds(samples.size());ds.upload(samples.data(),samples.size());
        std::vector<double> nodes,weights,x,w;
        for(int n:{8,16,32}) {
            gauss_legendre(n,x,w);nodes.insert(nodes.end(),x.begin(),x.end());weights.insert(weights.end(),w.begin(),w.end());
        }
        Buffer<double> dx(nodes.size()),dw(weights.size());dx.upload(nodes.data(),nodes.size());dw.upload(weights.data(),weights.size());
        auto run=[&](const std::vector<Rectangle>& batch,bool verify) {
            std::vector<Estimate> values(batch.size());
            if(batch.empty()) return values;
            Buffer<Rectangle> dr(batch.size());Buffer<Estimate> dv(batch.size());dr.upload(batch.data(),batch.size());
            rectangles<<<batch.size(),128>>>(p,ds.data,dr.data,dx.data,dw.data,verify,dv.data);
            check(gpuGetLastError());check(gpuDeviceSynchronize());
            check(gpuMemcpy(values.data(),dv.data,batch.size()*sizeof(Estimate),gpuMemcpyDeviceToHost));
            return values;
        };
        auto initial=run(tasks,false);size_t next=0;
        for(auto& list:leaves) for(auto& leaf:list) leaf.estimate=initial[next++];
        // Each live iteration either spends a checked budget or retires the
        // sample. Regions with nonfinite estimates never certify convergence.
        while(std::any_of(active.begin(),active.end(),[](int a){return a!=0;})) {
            std::vector<Rectangle> split,verify;
            std::vector<std::pair<size_t,size_t>> split_ids,verify_ids;
            for(size_t i=0;i<samples.size();++i) if(active[i]) {
                auto& list=leaves[i];double sum=0,err=0;bool finite=true,checked=true;
                for(const auto& leaf:list) {
                    const auto& e=leaf.estimate;
                    finite=finite && std::isfinite(e.value) && std::isfinite(e.error) && e.value>=0 && e.error>=0;
                    sum+=e.value;err+=e.error;checked=checked && leaf.verified;
                }
                refined[i].evaluations=result[i].evaluations+used[i];
                if(!finite || !std::isfinite(sum) || !std::isfinite(err)) { active[i]=0;continue; }
                refined[i].value=sum;refined[i].error=err;
                double threshold=std::max(1e-24,.1*tolerance*std::abs(sum));
                if(sum>0 && err<=threshold) {
                    if(checked) { refined[i].converged=true;active[i]=0;continue; }
                    size_t pending=std::count_if(list.begin(),list.end(),[](const Leaf& l){return !l.verified;});
                    if(pending>(budget-used[i])/(1024*cost_factor)) { active[i]=0;continue; }
                    used[i]+=unsigned(pending)*1024*cost_factor;
                    for(size_t j=0;j<list.size();++j) if(!list[j].verified) { verify.push_back(list[j].rect);verify_ids.push_back({i,j}); }
                    continue;
                }
                if(budget-used[i]<1152*cost_factor || !(sum>0)) { active[i]=0;continue; }
                auto worst=std::max_element(list.begin(),list.end(),[](const Leaf& a,const Leaf& b){return a.estimate.error<b.estimate.error;});
                size_t index=worst-list.begin();Rectangle a=worst->rect,b=a;
                bool polar=worst->estimate.polar_error>=worst->estimate.azimuth_error;
                double low=polar?a.u0:a.v0,high=polar?a.u1:a.v1,mid=(low+high)/2;
                if(!(mid>low && mid<high)) { active[i]=0;continue; }
                if(polar) { a.u1=mid;b.u0=mid; } else { a.v1=mid;b.v0=mid; }
                used[i]+=1152*cost_factor;
                list[index]={a,{},false};list.push_back({b,{},false});
                split.push_back(a);split_ids.push_back({i,index});
                split.push_back(b);split_ids.push_back({i,list.size()-1});
            }
            auto estimates=run(split,false);
            for(size_t k=0;k<split.size();++k) leaves[split_ids[k].first][split_ids[k].second].estimate=estimates[k];
            auto checks=run(verify,true);
            for(size_t k=0;k<verify.size();++k) {
                auto& leaf=leaves[verify_ids[k].first][verify_ids[k].second];
                leaf.estimate.error=std::max(leaf.estimate.error,std::abs(checks[k].value-leaf.estimate.value));
                leaf.estimate.value=checks[k].value;leaf.verified=true;
            }
        }
        result.swap(refined);return true;
    } catch(const GpuDeviceError& e) { error=e.what();if(device_error) *device_error=true;return false; }
    catch(const std::exception& e) { error=e.what();(void)gpuGetLastError();return false; }
}
bool EvaluateQelGpu(const QelParameters& p,const std::vector<QelSample>& samples,
                    const std::vector<double>& cosine,const std::vector<double>& phi,
                    int device,std::vector<double>& values,std::string& error,bool* device_error) {
    error.clear(); values.clear(); if(device_error) *device_error=false;
    if(samples.size()!=cosine.size() || samples.size()!=phi.size() || samples.size()>2147483647u) {
        error="Mismatched QEL query arrays"; return false;
    }
    if(samples.empty()) return true;
    try {
        check(gpuSetDevice(device)); const size_t n=samples.size();
        Buffer<QelSample> ds(n); Buffer<double> dc(n),dp(n),dv(n);
        ds.upload(samples.data(),n); dc.upload(cosine.data(),n); dp.upload(phi.data(),n);
        evaluate<<<(n+255)/256,256>>>(p,ds.data,dc.data,dp.data,n,dv.data);
        check(gpuGetLastError()); check(gpuDeviceSynchronize()); values.resize(n);
        check(gpuMemcpy(values.data(),dv.data,n*sizeof(double),gpuMemcpyDeviceToHost)); return true;
    } catch(const GpuDeviceError& e) { error=e.what(); if(device_error) *device_error=true; values.clear(); return false; }
    catch(const std::exception& e) { error=e.what(); (void)gpuGetLastError(); values.clear(); return false; }
}
// Overloads without the device_error flag (kept for existing callers).
bool IntegrateQelGpu(const QelParameters& p,const std::vector<QelSample>& samples,double tolerance,
                     unsigned budget,int device,std::vector<QelIntegralResult>& result,std::string& error) {
    return IntegrateQelGpu(p,samples,tolerance,budget,device,result,error,nullptr);
}
bool RefineQelGpu(const QelParameters& p,const std::vector<QelSample>& samples,double tolerance,
                  unsigned budget,int device,std::vector<QelIntegralResult>& result,std::string& error) {
    return RefineQelGpu(p,samples,tolerance,budget,device,result,error,nullptr);
}
bool EvaluateQelGpu(const QelParameters& p,const std::vector<QelSample>& samples,
                    const std::vector<double>& cosine,const std::vector<double>& phi,
                    int device,std::vector<double>& values,std::string& error) {
    return EvaluateQelGpu(p,samples,cosine,phi,device,values,error,nullptr);
}
}
