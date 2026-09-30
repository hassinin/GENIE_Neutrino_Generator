#ifndef GPU_SPLINE_QEL_INTEGRAND_H
#define GPU_SPLINE_QEL_INTEGRAND_H

// Llewellyn-Smith and Nieves NoRPA CCQE in GENIE's kPSQELEvGen phase space. No ROOT objects
// cross the device boundary. Nuclear sampling/binding and configuration stay
// with GENIE. See QELUtils.cxx and LwlynSmithQELCCPXSec.cxx for the reference.
#include "portable_gpu.h"
namespace gpu_spline {
struct QelVector {
    double x, y, z, t;
    GPU_HOST_DEVICE QelVector operator+(QelVector b) const { return {x+b.x,y+b.y,z+b.z,t+b.t}; }
    GPU_HOST_DEVICE QelVector operator-(QelVector b) const { return {x-b.x,y-b.y,z-b.z,t-b.t}; }
    GPU_HOST_DEVICE double dot(QelVector b) const { return t*b.t-x*b.x-y*b.y-z*b.z; }
    GPU_HOST_DEVICE double spatial2() const { return x*x+y*y+z*z; }
};
constexpr int kQelMaxZCoefficients=32;
struct QelParameters {
    QelVector probe;
    double initial_mass, final_mass, lepton_mass;
    double proton_mass, neutron_mass, pion_mass;
    double normalization; // GF^2 cos^2(thetaC) * nuclear count * xsec scale / (8 pi^2)
    double ma2, fa0, mu_p, mu_n;
    double gep_rational[4], gmp_rational[4]; // a1,b1,b2,b3
    double gep_nodes[7], gmp_nodes[7], gen_nodes[7], gmn_nodes[7];
    int antineutrino;
    int nieves, z_count;
    double z_t0, z_tcut, z_coefficients[kQelMaxZCoefficients];
};
struct QelSample {
    QelVector initial, on_shell, axis, transverse, normal;
    double gamma, beta, root_s, momentum, lepton_com, nucleon_com;
    double q2min, q2max, kf, cos_max;
    int bound, valid;
    double cos_min, coulomb, fermi_energy1, fermi_energy2;
};
GPU_HOST_DEVICE inline double qel_min(double a,double b) { return a<b?a:b; }
GPU_HOST_DEVICE inline double qel_max(double a,double b) { return a>b?a:b; }
GPU_HOST_DEVICE inline double qel_poly(double x,const double* c) {
    double sum=0;
    for(int i=0;i<7;++i) {
        double term=c[i], xi=i/6.;
        for(int j=0;j<7;++j) if(j!=i) term *= (x-j/6.)/(xi-j/6.);
        sum+=term;
    }
    return sum;
}
GPU_HOST_DEVICE inline double qel_rational(double tau,const double* c) {
    return (1+c[0]*tau)/(1+tau*(c[1]+tau*(c[2]+c[3]*tau)));
}
GPU_HOST_DEVICE inline void qel_form_factors(const QelParameters& p,double q2,
                                             double& f1,double& f2,double& fa,double& fp) {
    double tp=q2/(4*p.proton_mass*p.proton_mass), tn=q2/(4*p.neutron_mass*p.neutron_mass);
    double xp=2/(1+sqrt(1+1/tp)), xn=2/(1+sqrt(1+1/tn));
    double gep=qel_poly(xp,p.gep_nodes)*qel_rational(tp,p.gep_rational);
    double gmp=qel_poly(xp,p.gmp_nodes)*qel_rational(tp,p.gmp_rational)*p.mu_p;
    double gen=qel_poly(xn,p.gen_nodes)*gep*1.7*tn/(1+3.3*tn);
    double gmn=qel_poly(xn,p.gmn_nodes)*gmp*p.mu_n/p.mu_p;
    double tau=q2/(4*p.initial_mass*p.initial_mass);
    f1=(gep-gen+tau*(gmp-gmn))/(1+tau);
    f2=(gmp-gmn-gep+gen)/(1+tau);
    double d=1+q2/p.ma2;
    fa=p.fa0/(d*d);
    if(p.nieves) {
        double root=sqrt(p.z_tcut+q2), origin=sqrt(p.z_tcut-p.z_t0);
        double z=(root-origin)/(root+origin);
        fa=0;
        for(int i=0;i<p.z_count;++i) fa+=pow(z,i)*p.z_coefficients[i];
    }
    fp=2*p.initial_mass*p.initial_mass*fa/(p.pion_mass*p.pion_mass+q2);
}
// Azimuth is measured from the incident probe's projection perpendicular to
// the COM boost axis. Rotating the azimuthal origin leaves the integral invariant.
GPU_HOST_DEVICE inline void qel_final_state(const QelSample& s,double c,double phi,
                                            QelVector& lepton,QelVector& nucleon) {
    double sn=sqrt(qel_max(0.,1-c*c));
    double a=s.momentum*sn*cos(phi), b=s.momentum*sn*sin(phi);
    double z=s.gamma*(s.momentum*c+s.beta*s.lepton_com);
    lepton={a*s.transverse.x+b*s.normal.x+z*s.axis.x,
            a*s.transverse.y+b*s.normal.y+z*s.axis.y,
            a*s.transverse.z+b*s.normal.z+z*s.axis.z,
            s.gamma*(s.lepton_com+s.beta*s.momentum*c)};
    z=s.gamma*(-s.momentum*c+s.beta*s.nucleon_com);
    nucleon={-a*s.transverse.x-b*s.normal.x+z*s.axis.x,
             -a*s.transverse.y-b*s.normal.y+z*s.axis.y,
             -a*s.transverse.z-b*s.normal.z+z*s.axis.z,
             s.gamma*(s.nucleon_com-s.beta*s.momentum*c)};
}
#include "qel_nieves.h"
GPU_HOST_DEVICE inline double qel_integrand(const QelParameters& p,const QelSample& s,double c,double phi) {
    if(!s.valid || c>s.cos_max || (p.nieves && c<s.cos_min)) return 0;
    if(p.nieves) return qel_nieves_integrand(p,s,c,phi);
    QelVector l,n; qel_final_state(s,c,phi,l,n);
    if(n.spatial2()<s.kf*s.kf) return 0;
    QelVector q=p.probe-l, qt=n-s.on_shell;
    double Q2=-q.dot(q), Q2t=-qt.dot(qt);
    if(Q2<s.q2min || Q2>s.q2max || Q2t<=0 || (s.bound && qt.t<=0)) return 0;
    double f1,f2,fa,fp; qel_form_factors(p,Q2t,f1,f2,fa,fp);
    double mass=(p.initial_mass+p.final_mass)/2, m2=mass*mass, tau=Q2t/(4*m2);
    double h1=fa*fa*(1+tau)+tau*(f1+f2)*(f1+f2);
    double h2=fa*fa+f1*f1+tau*f2*f2;
    double h3=2*fa*(f1+f2);
    double h4=.25*f2*f2*(1-tau)+.5*f1*f2+fa*fp-tau*fp*fp;
    double kl=p.probe.dot(l), kp=p.probe.dot(s.on_shell), pl=s.on_shell.dot(l);
    double kq=p.probe.dot(qt), lq=l.dot(qt);
    double l1=2*kl*m2, l2=2*kp*pl-kl*m2;
    double l3=(p.antineutrino?1.:-1.)*(kp*lq-kq*pl);
    double l4=kl*qt.dot(qt)-2*kq*lq;
    double l5=kp*lq+pl*kq-kl*s.on_shell.dot(qt);
    double lh=2*(l1*h1+l2*h2+l3*h3+l4*h4+l5*h2);
    // EnergyDeltaFunctionSolutionQEL/(E_lepton E_nucleon) = p_COM/sqrt(s).
    return p.normalization*lh*s.momentum/(s.root_s*s.on_shell.t*p.probe.t);
}
GPU_HOST_DEVICE inline double qel_folded_integrand(const QelParameters& p,const QelSample& s,double c,double phi) {
    double value=qel_integrand(p,s,c,phi);
    return p.nieves ? .5*(value+qel_integrand(p,s,c,6.28318530717958647692-phi)) : value;
}
// Remove the sharp Q2 cuts analytically from the azimuthal integration.
// Q2=A-B*cos(phi), B>=0 with the chosen transverse axis. The second half
// of [0,2pi] has the same limits. The caller folds the integrand (explicitly
// averaging both halves for Nieves) and includes the factor of 2.
GPU_HOST_DEVICE inline bool qel_phi_limits(const QelParameters& p,const QelSample& s,
                                           double c,double& lo,double& hi) {
    QelVector l,n; qel_final_state(s,c,0,l,n);
    QelVector q=p.probe-l;
    double sn=sqrt(qel_max(0.,1-c*c));
    double kt=p.probe.x*s.transverse.x+p.probe.y*s.transverse.y+p.probe.z*s.transverse.z;
    double B=2*s.momentum*sn*qel_max(0.,kt), A=-q.dot(q)+B;
    double eps=s.on_shell.t-s.initial.t, q0=p.probe.t-l.t;
    double lower=p.nieves ? s.q2min : qel_max(s.q2min,eps*eps-2*eps*q0);
    if(B<=1e-30) { lo=0; hi=3.14159265358979323846; return A>=lower && A<=s.q2max; }
    double u=qel_min(1.,(A-lower)/B), v=qel_max(-1.,(A-s.q2max)/B);
    if(p.nieves) {
        double local=l.t-(p.antineutrino?-1.:1.)*s.coulomb;
        if(local<=p.lepton_mass || l.t-p.lepton_mass<=fabs(s.coulomb)) return false;
        double ratio=sqrt((local*local-p.lepton_mass*p.lepton_mass)/l.spatial2());
        QelVector qt{p.probe.x-ratio*l.x,p.probe.y-ratio*l.y,p.probe.z-ratio*l.z,n.t-s.on_shell.t};
        double D=ratio*B,C=qt.spatial2()+D;
        double low=sqrt(qel_max(0.,C-D)),high=sqrt(qel_max(0.,C+D));
        double original_low=low,original_high=high;
        if(!qel_nieves_transfer_limits(p,s,qt.t,low,high)) return false;
        // Do not round-trip an unchanged endpoint through sqrt/square/acos:
        // that can manufacture a small cut at phi=0 or pi. Round new bounds
        // outward as well; the native integrand still enforces every cut.
        const double roundoff=16*2.2204460492503131e-16;
        if(low>original_low) u=qel_min(u,(C-low*low+roundoff*(fabs(C)+low*low))/D);
        if(high<original_high) v=qel_max(v,(C-high*high-roundoff*(fabs(C)+high*high))/D);
    }
    if(u<=v) return false;
    lo=acos(qel_max(-1.,u)); hi=acos(qel_min(1.,v));
    return hi>lo;
}
} // namespace gpu_spline
#endif
