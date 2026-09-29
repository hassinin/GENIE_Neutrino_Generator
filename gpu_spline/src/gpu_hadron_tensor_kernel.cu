#include "gpu_spline/gpu_hadron_tensor.h"
#include "gpu_spline/tensor_kinematics.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <limits>
#include <cstdint>

#if defined(GPU_ENABLE_CUDA) || defined(GPU_ENABLE_HIP)
#include <curand_kernel.h>
#endif

namespace gpuspline {

GPU_DEVICE inline double gpu_real_sqrt(double x) {
    return (x <= 0.0) ? 0.0 : sqrt(x);
}

GPU_DEVICE inline GpuHadronTensorEntry gpu_hadron_tensor_interpolate(
    const GpuHadronTensorData& tensor, double q0, double q3)
{
    double eval_q0 = (q0 < tensor.q0_min) ? tensor.q0_min : ((q0 > tensor.q0_max) ? tensor.q0_max : q0);
    double eval_q3 = (q3 < tensor.q3_min) ? tensor.q3_min : ((q3 > tensor.q3_max) ? tensor.q3_max : q3);

    int ix_lo = 0, ix_hi = 0;
    int iy_lo = 0, iy_hi = 0;

    // x = q0
    if (tensor.q0_flag == 0 && tensor.q0_step > 0.0) {
        ix_lo = (int)((eval_q0 - tensor.q0_start) / tensor.q0_step);
        if (ix_lo < 0) ix_lo = 0;
        if (ix_lo > tensor.num_q0 - 2) ix_lo = tensor.num_q0 - 2;
        ix_hi = ix_lo + 1;
    } else {
        // Binary search
        int low = 0, high = tensor.num_q0 - 1;
        while (low < high) {
            int mid = (low + high) / 2;
            if (tensor.d_q0_points[mid] < eval_q0) low = mid + 1;
            else high = mid;
        }
        ix_lo = (low > 0) ? low - 1 : 0;
        if (ix_lo > tensor.num_q0 - 2) ix_lo = tensor.num_q0 - 2;
        ix_hi = ix_lo + 1;
    }

    // y = q3
    if (tensor.q3_flag == 0 && tensor.q3_step > 0.0) {
        iy_lo = (int)((eval_q3 - tensor.q3_start) / tensor.q3_step);
        if (iy_lo < 0) iy_lo = 0;
        if (iy_lo > tensor.num_q3 - 2) iy_lo = tensor.num_q3 - 2;
        iy_hi = iy_lo + 1;
    } else {
        // Binary search
        int low = 0, high = tensor.num_q3 - 1;
        while (low < high) {
            int mid = (low + high) / 2;
            if (tensor.d_q3_points[mid] < eval_q3) low = mid + 1;
            else high = mid;
        }
        iy_lo = (low > 0) ? low - 1 : 0;
        if (iy_lo > tensor.num_q3 - 2) iy_lo = tensor.num_q3 - 2;
        iy_hi = iy_lo + 1;
    }

    double x1 = (tensor.q0_flag == 0) ? (tensor.q0_start + ix_lo * tensor.q0_step) : tensor.d_q0_points[ix_lo];
    double x2 = (tensor.q0_flag == 0) ? (tensor.q0_start + ix_hi * tensor.q0_step) : tensor.d_q0_points[ix_hi];
    double y1 = (tensor.q3_flag == 0) ? (tensor.q3_start + iy_lo * tensor.q3_step) : tensor.d_q3_points[iy_lo];
    double y2 = (tensor.q3_flag == 0) ? (tensor.q3_start + iy_hi * tensor.q3_step) : tensor.d_q3_points[iy_hi];

    const GpuHadronTensorEntry& z11 = tensor.d_entries[ix_lo * tensor.num_q3 + iy_lo];
    const GpuHadronTensorEntry& z21 = tensor.d_entries[ix_hi * tensor.num_q3 + iy_lo];
    const GpuHadronTensorEntry& z12 = tensor.d_entries[ix_lo * tensor.num_q3 + iy_hi];
    const GpuHadronTensorEntry& z22 = tensor.d_entries[ix_hi * tensor.num_q3 + iy_hi];

    double dy = y2 - y1;
    double dx = x2 - x1;
    double wy1 = (dy > 0.0) ? (y2 - eval_q3) / dy : 1.0;
    double wy2 = (dy > 0.0) ? (eval_q3 - y1) / dy : 0.0;
    double wx1 = (dx > 0.0) ? (x2 - eval_q0) / dx : 1.0;
    double wx2 = (dx > 0.0) ? (eval_q0 - x1) / dx : 0.0;

    GpuHadronTensorEntry z1 = z11 * wy1 + z12 * wy2;
    GpuHadronTensorEntry z2 = z21 * wy1 + z22 * wy2;
    return z1 * wx1 + z2 * wx2;
}

GPU_DEVICE inline double gpu_hadron_tensor_diff_xsec(
    const GpuHadronTensorData& tensor,
    int probe_pdg, double E_probe, double m_probe,
    double Tl, double cos_l, double ml, double Q_value,
    bool use_rosenbluth, double Vud)
{
    double El = Tl + ml;
    double q0 = E_probe - El;
    double q0_corrected = q0 - Q_value;

    double k_initial = gpu_real_sqrt(E_probe * E_probe - m_probe * m_probe);
    double k_final = gpu_real_sqrt(Tl * Tl + 2.0 * ml * Tl);
    double q_mag2 = TensorQ3Squared(k_initial, k_final, cos_l);
    double q_mag = gpu_real_sqrt(q_mag2);

    if (!(q_mag2 > 0.0) || !TensorInSupport(q0_corrected, q_mag, E_probe,
            tensor.q0_min, tensor.q0_max, tensor.q3_min, tensor.q3_max)) {
        return 0.0;
    }

    double q2 = q0 * q0 - q_mag2;
    double Q2 = q_mag2 - q0 * q0;
    if (Q2 < 0.0) return 0.0;

    GpuHadronTensorEntry entry = gpu_hadron_tensor_interpolate(tensor, q0_corrected, q_mag);

    const double kPi = 3.14159265358979323846;
    const double kGF = 1.16639E-5;
    const double kGF2 = kGF * kGF;

    double s2_half = (1.0 - cos_l) * 0.5;
    double c2_half = 1.0 - s2_half;

    if (probe_pdg == 11) {
        // GENIE has distinct electron contractions. Only Rosenbluth applies
        // the finite-mass Mott factor and the tensor Wxx / 2 correction.
        if (Q2 <= 0.0) return 0.0;
        const double alpha = 1.0 / 137.03599976;
        const double mott_amp = use_rosenbluth ? 2.0 * El * alpha / q2 :
                                               alpha / (2.0 * E_probe * s2_half);
        const double longitudinal = (q2 / q_mag2) * (q2 / q_mag2) * entry.W00 * c2_half;
        const double transverse = (2.0 * s2_half - q2 * c2_half / q_mag2) * entry.Wxx;
        return 2.0 * kPi * mott_amp * mott_amp *
            (longitudinal + (use_rosenbluth ? transverse / 2.0 : transverse));
    }
    const int abs_pdg = (probe_pdg < 0) ? -probe_pdg : probe_pdg;
    if (abs_pdg != 12 && abs_pdg != 14 && abs_pdg != 16) return 0.0;

    if (!use_rosenbluth) {
        // Standard Valencia formulation
        double w1 = entry.Wxx * 0.5;
        double temp_1 = (q0 * q0 / q_mag2) * (entry.Wzz - entry.Wxx);
        double temp_2 = 2.0 * q0 * entry.ReW0z / q_mag;
        double w2 = (entry.W00 + entry.Wxx + temp_1 - temp_2) * 0.5;

        double w3 = entry.ImWxy / q_mag;
        if (probe_pdg < 0) w3 *= -1.0;

        double w4 = (entry.Wzz - entry.Wxx) / (2.0 * q_mag2);
        double w5 = (entry.ReW0z - q0 * (entry.Wzz - entry.Wxx) / q_mag) / q_mag;

        double part_1 = (2.0 * w1 * s2_half) + (w2 * c2_half) - (w3 * (E_probe + El) * s2_half);
        double part_2 = (w1 * cos_l) - (w2 * 0.5 * cos_l)
                      + (w3 * 0.5 * (El + k_final - (E_probe + El) * cos_l))
                      + (w4 * 0.5 * (ml * ml * cos_l + 2.0 * El * (El + k_final) * s2_half))
                      - (w5 * 0.5 * (El + k_final));

        double denom = El * (El + k_final);
        double all_terms = part_1 + ((denom > 0.0) ? (ml * ml * part_2 / denom) : 0.0);

        return (2.0 / kPi) * k_final * El * kGF2 * all_terms;
    } else {
        // SuSAv2 Rosenbluth formulation
        double v0 = 4.0 * El * E_probe + q2;
        double Vud2 = Vud * Vud;
        double sig0 = kGF2 * Vud2 * v0 * k_final / (4.0 * kPi * E_probe);

        double neg_q2 = -q2;
        if (neg_q2 <= 0.0) return 0.0;
        double sqrt_neg_q2 = sqrt(neg_q2);

        double xdelta = ml / sqrt_neg_q2;
        double xrho = neg_q2 / q_mag2;
        double xrhop = q_mag / (El + E_probe);
        double tan2th2 = neg_q2 / v0;

        double xdelta2 = xdelta * xdelta;
        double VCC = 1.0 - xdelta2 * tan2th2;
        double VCL = q0 / q_mag + tan2th2 * xdelta2 / xrhop;
        double VLL = (q0 * q0) / q_mag2 + xdelta2 * tan2th2 * (1.0 + 2.0 * q0 / q_mag / xrhop + xrho * xdelta2);
        double VT = xrho * 0.5 + tan2th2 - tan2th2 * xdelta2 / xrhop * (q0 / q_mag + xdelta2 * xrho * xrhop * 0.5);
        double VTP = tan2th2 / xrhop * (1.0 - q0 / q_mag * xrhop * xdelta2);

        double RCC = entry.W00;
        double RCL = -entry.ReW0z;
        double RLL = entry.Wzz;
        double RT = 2.0 * entry.Wxx;
        double RTP = -entry.ImWxy;
        if (probe_pdg < 0) RTP *= -1.0;

        double xsec = sig0 * (VCC * RCC + 2.0 * VCL * RCL + VLL * RLL + VT * RT + 2.0 * VTP * RTP);
        return (xsec < 0.0) ? 0.0 : xsec;
    }
}

// Global batch evaluation kernel
GPU_GLOBAL void k_eval_hadron_tensor_diff_xsec(
    GpuHadronTensorData tensor,
    const MecQuery* queries,
    double* out_xsec,
    int num_queries,
    bool use_rosenbluth,
    double Vud)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_queries) {
        const MecQuery& q = queries[tid];
        out_xsec[tid] = gpu_hadron_tensor_diff_xsec(
            tensor, q.probe_pdg, q.E_probe, q.m_probe,
            q.Tl, q.cos_l, q.ml, q.Q_value,
            use_rosenbluth, Vud);
    }
}

#if defined(GPU_ENABLE_CUDA) || defined(GPU_ENABLE_HIP)
GPU_DEVICE inline void atomicMaxDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        if (__longlong_as_double(assumed) >= val) break;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val));
    } while (assumed != old);
}

// 2D Phase Space Max XSec Search Kernel
GPU_GLOBAL void k_find_max_xsec(
    GpuHadronTensorData tensor,
    int probe_pdg, double E_probe, double m_probe, double ml,
    double T_min, double T_max, double costh_min, double costh_max,
    double Q_value, bool use_rosenbluth, double Vud,
    double* d_max_xsec,
    int n_steps_T, int n_steps_ctl)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_points = n_steps_T * n_steps_ctl;
    if (tid >= total_points) return;

    int i_T = tid / n_steps_ctl;
    int i_ctl = tid % n_steps_ctl;

    double T = T_min + (T_max - T_min) * ((double)i_T / (double)(n_steps_T - 1));
    double costh = costh_min + (costh_max - costh_min) * ((double)i_ctl / (double)(n_steps_ctl - 1));

    double xsec = gpu_hadron_tensor_diff_xsec(
        tensor, probe_pdg, E_probe, m_probe, T, costh, ml, Q_value,
        use_rosenbluth, Vud);

    __shared__ double s_max[256];
    s_max[threadIdx.x] = xsec;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (s_max[threadIdx.x + s] > s_max[threadIdx.x]) {
                s_max[threadIdx.x] = s_max[threadIdx.x + s];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicMaxDouble(d_max_xsec, s_max[0]);
    }
}

GPU_DEVICE inline double tensor_scan_coordinate(const GpuHadronTensorData& t, bool q0, int half_index) {
    const int node = half_index / 2;
    const int flag = q0 ? t.q0_flag : t.q3_flag;
    const double start = q0 ? t.q0_start : t.q3_start;
    const double step = q0 ? t.q0_step : t.q3_step;
    const double* points = q0 ? t.d_q0_points : t.d_q3_points;
    const double a = flag == 0 ? start + node*step : points[node];
    if (!(half_index % 2)) return a;
    const double b = flag == 0 ? start + (node+1)*step : points[node+1];
    return (a+b)*0.5;
}

// A uniform (T,cos theta) grid misses narrow forward ridges. Scan in the
// coordinates of the tabulated function instead, before drawing any random
// proposals. Project grid points onto the requested lepton-domain edges.
GPU_GLOBAL void k_find_table_max_xsec(
    GpuHadronTensorData tensor, int probe, double E, double mi, double ml,
    double Tmin, double Tmax, double cmin, double cmax, double Q, bool rb, double Vud,
    double Q3Max, double Q2min, int first0, int first3, int n0, int n3, double* maximum)
{
    const int tid = blockIdx.x*blockDim.x + threadIdx.x;
    double value = 0.0;
    if (tid < n0*n3) {
        const double q0 = tensor_scan_coordinate(tensor, true, first0+tid/n3);
        const double q3 = tensor_scan_coordinate(tensor, false, first3+tid%n3);
        const double T = fmax(Tmin, fmin(Tmax, E-ml-Q-q0));
        const double ki = gpu_real_sqrt(E*E-mi*mi);
        const double kf = gpu_real_sqrt(T*(T+2.0*ml));
        if (ki > 0.0 && kf > 0.0) {
            const double c = fmax(cmin, fmin(cmax,
                1.0-((q3*q3-(ki-kf)*(ki-kf))/(2.0*ki*kf))));
            // Use the same cut arithmetic as the rejection kernel.
            const double omega = E-(T+ml);
            const double cut_q3 = gpu_real_sqrt(kf*kf+E*E-2.0*kf*E*c);
            if (cut_q3 < Q3Max && cut_q3*cut_q3-omega*omega >= Q2min) {
                value = gpu_hadron_tensor_diff_xsec(tensor,probe,E,mi,T,c,ml,Q,rb,Vud);
                if (!isfinite(value)) value = INFINITY;
            }
        }
    }
    __shared__ double values[256];
    values[threadIdx.x] = value;
    __syncthreads();
    for (int offset=blockDim.x/2; offset>0; offset/=2) {
        if (threadIdx.x<offset) values[threadIdx.x]=fmax(values[threadIdx.x],values[threadIdx.x+offset]);
        __syncthreads();
    }
    if (threadIdx.x==0) atomicMaxDouble(maximum,values[0]);
}

// Kernel for rejection sampling candidate kinematics
GPU_GLOBAL void k_sample_mec_kinematics(
    GpuHadronTensorData tensor,
    int probe_pdg, double E_probe, double m_probe, double ml,
    double T_min, double T_max, double costh_min, double costh_max,
    double Q_value, double xsec_max, bool use_rosenbluth, double Vud,
    unsigned long long seed,
    MecKinematicsCandidate* out_candidates,
    int* accepted_count,
    int batch_size, double Q3Max, double Q2min, double* bound_exceeded)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, tid, 0, &state);

    float4 r = curand_uniform4(&state);
    double T = T_min + (T_max - T_min) * (double)r.x;
    double costh = costh_min + (costh_max - costh_min) * (double)r.y;
    double rnd_accept = (double)r.z;

    // Match GENIE's phase-space cuts before applying the supplied bound.
    const double q0 = E_probe - (T + ml);
    const double p = gpu_real_sqrt(T * (T + 2.0 * ml));
    const double q3 = gpu_real_sqrt(p*p + E_probe*E_probe - 2.0*p*E_probe*costh);
    if (q3 >= Q3Max || q3*q3 - q0*q0 < Q2min) return;

    double xsec = gpu_hadron_tensor_diff_xsec(
        tensor, probe_pdg, E_probe, m_probe, T, costh, ml, Q_value,
        use_rosenbluth, Vud);

    if (!isfinite(xsec) || xsec > xsec_max) {
        atomicMaxDouble(bound_exceeded, 1.0);
        return;
    }
    bool accepted = (xsec > 0.0) && (xsec >= xsec_max * rnd_accept);

    if (accepted) {
        // Select by proposal index, not by nondeterministic warp arrival order.
        out_candidates[tid] = MecKinematicsCandidate{};
        out_candidates[tid].Tl = T;
        out_candidates[tid].cos_l = costh;
        out_candidates[tid].xsec = xsec;
        out_candidates[tid].accepted = true;
        atomicMin(accepted_count, tid);
    }
}
#endif

double GpuHadronTensor::FindMaxXSecGPU(
    int probe_pdg, double E_probe, double m_probe, double ml,
    double T_min, double T_max, double costh_min, double costh_max,
    double Q_value, bool use_rosenbluth, double Vud,
    double safety_factor) const
{
#if defined(GPU_ENABLE_CUDA) || defined(GPU_ENABLE_HIP)
    if (!fIsOnDevice || !d_max_xsec) return 0.0;
    gpuSetDevice(fDeviceId);

    double zero = 0.0;
    GPU_CHECK(gpuMemcpy(d_max_xsec, &zero, sizeof(double), gpuMemcpyHostToDevice));

    int n_steps_T = 64;
    int n_steps_ctl = 64;
    int total_points = n_steps_T * n_steps_ctl; // 4096 points
    int block_size = 256;
    int grid_size = (total_points + block_size - 1) / block_size;

    k_find_max_xsec<<<grid_size, block_size>>>(
        fHeader, probe_pdg, E_probe, m_probe, ml,
        T_min, T_max, costh_min, costh_max, Q_value,
        use_rosenbluth, Vud, d_max_xsec, n_steps_T, n_steps_ctl);

    GPU_CHECK(gpuDeviceSynchronize());

    double h_max_xsec = 0.0;
    GPU_CHECK(gpuMemcpy(&h_max_xsec, d_max_xsec, sizeof(double), gpuMemcpyDeviceToHost));

    return h_max_xsec * safety_factor;
#else
    return 0.0;
#endif
}

void GpuHadronTensor::EvalDiffXSecBatch(
    const MecQuery* queries, double* out_xsec, int num_queries,
    bool use_rosenbluth, double Vud) const
{
    if (num_queries <= 0) return;
    // Loading/replacing a table invalidates the device copy. Host-input calls
    // remain usable with the current table until it is uploaded again.
    if (!fIsOnDevice) {
        for (int i = 0; i < num_queries; ++i) {
            const MecQuery& q = queries[i];
            out_xsec[i] = DiffXSecHost(q.probe_pdg, q.E_probe, q.m_probe,
                q.Tl, q.cos_l, q.ml, q.Q_value, use_rosenbluth, Vud);
        }
        return;
    }

#if defined(GPU_ENABLE_CUDA) || defined(GPU_ENABLE_HIP)
    gpuSetDevice(fDeviceId);

    MecQuery* d_queries = nullptr;
    double* d_out_xsec = nullptr;
    size_t q_bytes = num_queries * sizeof(MecQuery);
    size_t out_bytes = num_queries * sizeof(double);

    GPU_CHECK(gpuMalloc((void**)&d_queries, q_bytes));
    GPU_CHECK(gpuMalloc((void**)&d_out_xsec, out_bytes));

    GPU_CHECK(gpuMemcpy(d_queries, queries, q_bytes, gpuMemcpyHostToDevice));

    int block_size = 256;
    int grid_size = (num_queries + block_size - 1) / block_size;

    k_eval_hadron_tensor_diff_xsec<<<grid_size, block_size>>>(
        fHeader, d_queries, d_out_xsec, num_queries, use_rosenbluth, Vud);

    GPU_CHECK(gpuDeviceSynchronize());
    GPU_CHECK(gpuMemcpy(out_xsec, d_out_xsec, out_bytes, gpuMemcpyDeviceToHost));

    gpuFree(d_queries);
    gpuFree(d_out_xsec);
#else
    for (int i = 0; i < num_queries; ++i) {
        const MecQuery& q = queries[i];
        out_xsec[i] = DiffXSecHost(
            q.probe_pdg, q.E_probe, q.m_probe,
            q.Tl, q.cos_l, q.ml, q.Q_value,
            use_rosenbluth, Vud);
    }
#endif
}

void GpuHadronTensor::EvalDiffXSecDevice(
    const MecQuery* d_queries, double* d_out_xsec, int num_queries,
    bool use_rosenbluth, double Vud) const
{
    if (num_queries <= 0) return;
    if (!fIsOnDevice) {
        throw std::runtime_error("Upload the current hadron tensor before device-input evaluation");
    }
#if defined(GPU_ENABLE_CUDA) || defined(GPU_ENABLE_HIP)
    gpuSetDevice(fDeviceId);
    int block_size = 256;
    int grid_size = (num_queries + block_size - 1) / block_size;
    k_eval_hadron_tensor_diff_xsec<<<grid_size, block_size>>>(
        fHeader, d_queries, d_out_xsec, num_queries, use_rosenbluth, Vud);
    GPU_CHECK(gpuDeviceSynchronize());
#endif
}

double GpuHadronTensor::FindMaxXSecOnTableGPU(
    int probe, double E, double mi, double ml, double Tmin, double Tmax,
    double cmin, double cmax, double Q, bool rb, double Vud,
    double Q3Max, double Q2min, double safety_factor) const
{
#if defined(GPU_ENABLE_CUDA) || defined(GPU_ENABLE_HIP)
    if (!fIsOnDevice || !d_max_xsec || Tmax<=Tmin || cmax<=cmin) return 0.0;
    if (!(safety_factor>=1.0) || !std::isfinite(safety_factor))
        throw std::invalid_argument("Invalid table-scan safety factor");
    // Include bracketing nodes even when a kinematic edge cuts a table cell.
    auto bounds = [](const std::vector<double>& points, double lo, double hi) {
        int first = std::max(0, int(std::lower_bound(points.begin(),points.end(),lo)-points.begin())-1);
        int last = std::min(int(points.size())-1, int(std::lower_bound(points.begin(),points.end(),hi)-points.begin()));
        return std::pair<int,int>(2*first, 2*last);
    };
    const auto a=bounds(fHostQ0,E-ml-Q-Tmax,E-ml-Q-Tmin);
    const auto b=bounds(fHostQ3,0.0,std::min(Q3Max,2.0*E));
    const int n0=a.second-a.first+1, n3=b.second-b.first+1;
    if (n0<=0 || n3<=0) return 0.0;
    const std::int64_t total=std::int64_t(n0)*n3;
    if (total>std::numeric_limits<int>::max()-255) throw std::overflow_error("Tensor scan grid too large");
    GPU_CHECK(gpuSetDevice(fDeviceId));
    double result=0.0;
    GPU_CHECK(gpuMemcpy(d_max_xsec,&result,sizeof(double),gpuMemcpyHostToDevice));
    k_find_table_max_xsec<<<(int(total)+255)/256,256>>>(fHeader,probe,E,mi,ml,Tmin,Tmax,cmin,cmax,
        Q,rb,Vud,Q3Max,Q2min,a.first,b.first,n0,n3,d_max_xsec);
    GPU_CHECK(gpuDeviceSynchronize());
    GPU_CHECK(gpuMemcpy(&result,d_max_xsec,sizeof(double),gpuMemcpyDeviceToHost));
    if (!std::isfinite(result)) throw std::runtime_error("Nonfinite tensor value in deterministic envelope scan");
    return result*safety_factor;
#else
    return 0.0;
#endif
}

bool GpuHadronTensor::SampleKinematicsGPU(
    int probe_pdg, double E_probe, double m_probe, double ml,
    double T_min, double T_max, double costh_min, double costh_max,
    double Q_value, double xsec_max, bool use_rosenbluth,
    double Vud, unsigned long long seed,
    MecKinematicsCandidate& out_result, int batch_size,
    double Q3Max, double Q2min) const
{
    out_result = MecKinematicsCandidate{};
    if (!(xsec_max > 0.0) || !std::isfinite(xsec_max)) {
        throw std::invalid_argument("GPU sampling requires a finite positive cross-section upper bound");
    }
#if defined(GPU_ENABLE_CUDA) || defined(GPU_ENABLE_HIP)
    if (!fIsOnDevice || !d_sample_candidates || !d_sample_count) return false;
    GPU_CHECK(gpuSetDevice(fDeviceId));
    batch_size = std::min(batch_size, fSampleBatchSize);
    if (batch_size <= 0 || T_max <= T_min || costh_max <= costh_min) return false;

    const int block_size = 256;
    const int grid_size = (batch_size + block_size - 1) / block_size;
    for (int attempt = 0; attempt < 2; ++attempt) {
        int first_accepted = batch_size;
        double bound_exceeded = 0.0;
        GPU_CHECK(gpuMemcpy(d_sample_count, &first_accepted, sizeof(int), gpuMemcpyHostToDevice));
        GPU_CHECK(gpuMemcpy(d_max_xsec, &bound_exceeded, sizeof(double), gpuMemcpyHostToDevice));
        k_sample_mec_kinematics<<<grid_size, block_size>>>(
            fHeader, probe_pdg, E_probe, m_probe, ml,
            T_min, T_max, costh_min, costh_max, Q_value, xsec_max,
            use_rosenbluth, Vud, seed + attempt * 1234567ULL,
            d_sample_candidates, d_sample_count, batch_size, Q3Max, Q2min, d_max_xsec);
        GPU_CHECK(gpuDeviceSynchronize());
        GPU_CHECK(gpuMemcpy(&bound_exceeded, d_max_xsec, sizeof(double), gpuMemcpyDeviceToHost));
        if (bound_exceeded != 0.0) {
            // Do not retry/fall back conditionally: that can also bias sampling.
            throw std::runtime_error("GPU rejection bound exceeded: increase the supplied cross-section bound before generating events");
        }
        GPU_CHECK(gpuMemcpy(&first_accepted, d_sample_count, sizeof(int), gpuMemcpyDeviceToHost));
        if (first_accepted < batch_size) {
            GPU_CHECK(gpuMemcpy(&out_result, d_sample_candidates + first_accepted,
                                sizeof(MecKinematicsCandidate), gpuMemcpyDeviceToHost));
            return true;
        }
    }
#endif
    return false;
}

#if defined(GPU_ENABLE_CUDA) || defined(GPU_ENABLE_HIP)
GPU_GLOBAL void k_integrate_hadron_tensor_2d(
    GpuHadronTensorData tensor,
    int probe_pdg, double E_probe, double m_probe, double ml,
    double Delta_Q_value,
    double T_min, double T_max, double costh_min, double costh_max,
    double Q3Max, double Q2min,
    bool use_rosenbluth, double Vud,
    double* d_out_integral,
    int n_points_T, int n_points_costh)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_points = n_points_T * n_points_costh;

    double sum = 0.0;
    if (idx < total_points) {
        int i_T = idx / n_points_costh;
        int j_ctl = idx % n_points_costh;

        double w_T = (i_T == 0 || i_T == n_points_T - 1) ? 1.0 : ((i_T % 2 == 1) ? 4.0 : 2.0);
        double w_ctl = (j_ctl == 0 || j_ctl == n_points_costh - 1) ? 1.0 : ((j_ctl % 2 == 1) ? 4.0 : 2.0);

        double Tl = T_min + (double)i_T * (T_max - T_min) / (double)(n_points_T - 1);
        double ctl = costh_min + (double)j_ctl * (costh_max - costh_min) / (double)(n_points_costh - 1);

        double El = Tl + ml;
        double q0 = E_probe - El;
        double Plep = gpu_real_sqrt(Tl * (Tl + 2.0 * ml));
        double Q3 = gpu_real_sqrt(Plep * Plep + E_probe * E_probe - 2.0 * Plep * E_probe * ctl);
        double Q2 = Q3 * Q3 - q0 * q0;

        if (Q3 < Q3Max && Q2 >= Q2min) {
            double d2xsec = gpu_hadron_tensor_diff_xsec(
                tensor, probe_pdg, E_probe, m_probe, Tl, ctl, ml,
                Delta_Q_value, use_rosenbluth, Vud);
            sum = w_T * w_ctl * d2xsec;
        }
    }

    __shared__ double sdata[256];
    sdata[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0 && sdata[0] != 0.0) {
        atomicAdd(d_out_integral, sdata[0]);
    }
}
#endif

bool GpuHadronTensor::Integrate2DGPU(
    int probe_pdg, double E_probe, double m_probe, double ml,
    double Delta_Q_value,
    double T_min, double T_max, double costh_min, double costh_max,
    double Q3Max, double Q2min,
    bool use_rosenbluth, double Vud,
    double& out_total_xsec,
    int n_points_T, int n_points_costh) const
{
    out_total_xsec = 0.0;
    // Simpson quadrature needs at least two intervals per axis. Check before
    // rounding or launching, including the signed indexing used by the kernel.
    if (n_points_T < 3 || n_points_costh < 3) return false;
    if (n_points_T % 2 == 0) n_points_T++;
    if (n_points_costh % 2 == 0) n_points_costh++;
    const int block_size = 256;
    const std::int64_t count = static_cast<std::int64_t>(n_points_T) * n_points_costh;
    if (count > std::numeric_limits<int>::max() - (block_size - 1)) return false;
#if defined(GPU_ENABLE_CUDA) || defined(GPU_ENABLE_HIP)
    if (!fIsOnDevice || !d_integral) return false;
    if (T_max <= T_min || costh_max <= costh_min) {
        out_total_xsec = 0.0;
        return true;
    }
    gpuSetDevice(fDeviceId);

    double zero = 0.0;
    GPU_CHECK(gpuMemcpy(d_integral, &zero, sizeof(double), gpuMemcpyHostToDevice));

    int total_points = static_cast<int>(count);
    int grid_size = (total_points + block_size - 1) / block_size;

    k_integrate_hadron_tensor_2d<<<grid_size, block_size>>>(
        fHeader, probe_pdg, E_probe, m_probe, ml,
        Delta_Q_value, T_min, T_max, costh_min, costh_max,
        Q3Max, Q2min, use_rosenbluth, Vud,
        d_integral, n_points_T, n_points_costh);

    GPU_CHECK(gpuDeviceSynchronize());

    double raw_sum = 0.0;
    GPU_CHECK(gpuMemcpy(&raw_sum, d_integral, sizeof(double), gpuMemcpyDeviceToHost));

    double h_T = (T_max - T_min) / (double)(n_points_T - 1);
    double h_ctl = (costh_max - costh_min) / (double)(n_points_costh - 1);
    out_total_xsec = raw_sum * (h_T / 3.0) * (h_ctl / 3.0);
    if (!std::isfinite(out_total_xsec)) {
        out_total_xsec = 0.0;
        return false;
    }
    return true;
#else
    out_total_xsec = 0.0;
    return false;
#endif
}

} // namespace gpuspline
