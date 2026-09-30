#ifndef GPU_SPLINE_QEL_NIEVES_H
#define GPU_SPLINE_QEL_NIEVES_H
// Included inside gpu_spline after the common QEL data and form-factor helpers.
// Exact NoRPA tensor components from NievesQELCCPXSec::LmunuAnumu.
GPU_HOST_DEVICE inline QelVector qel_rotate_to_z(QelVector v,QelVector q) {
    double perp=sqrt(q.x*q.x+q.y*q.y), mag=sqrt(q.spatial2());
    if(perp==0) return q.z<0 ? QelVector{-v.x,v.y,-v.z,v.t} : v;
    double ax=q.y/perp, ay=-q.x/perp;
    double angle=acos(qel_max(-1.,qel_min(1.,q.z/mag)));
    double cs=cos(angle),sn=sin(angle),dot=ax*v.x+ay*v.y;
    return {v.x*cs+ay*v.z*sn+ax*dot*(1-cs),
            v.y*cs-ax*v.z*sn+ay*dot*(1-cs),
            v.z*cs+(ax*v.y-ay*v.x)*sn,v.t};
}
// Bound the momentum transfer accepted by the native NoRPA Lindhard cut.
// For q0>0, a(d) has its minimum M at d=sqrt(q0^2+2*M*q0).
// The native imU <= 1e-6 condition is a(d) <= EF1 + epsilon*d,
// together with the two constant terms in max(M, EF2-q0, a).
// Keep the rejected side of each bracket: rounding may include an extra zero
// sliver, but must not cut away positive integrand values.
GPU_HOST_DEVICE inline double qel_nieves_lindhard_edge(double d,double q0,double m,double ef,double epsilon) {
    double spacelike=d*d-q0*q0;
    if(spacelike<=0) return 1e300;
    return .5*(-q0+d*sqrt(1+4*m*m/spacelike))-ef-epsilon*d;
}
GPU_HOST_DEVICE inline bool qel_nieves_transfer_limits(const QelParameters& p,const QelSample& s,
                                                       double q0,double& low,double& high) {
    low=qel_max(low,sqrt(q0*q0+1e-6));
    if(!(high>low)) return false;
    if(!s.bound) return true;
    if(!(q0>0)) return false;
    double m=(p.initial_mass+p.final_mass)/2;
    double epsilon=6.28318530717958647692*1e-6/(m*m);
    double center=sqrt(q0*q0+2*m*q0);
    double a=q0,b=center;
    if(low<center && qel_nieves_lindhard_edge(low,q0,m,s.fermi_energy1,epsilon)>0) {
        a=low;
        for(int it=0;it<48;++it) {
            double mid=(a+b)/2;
            if(qel_nieves_lindhard_edge(mid,q0,m,s.fermi_energy1,epsilon)>0) a=mid; else b=mid;
        }
        low=qel_max(low,a);
    }
    a=center;b=qel_max(center,high);
    if(qel_nieves_lindhard_edge(b,q0,m,s.fermi_energy1,epsilon)>0) {
        for(int it=0;it<48;++it) {
            double mid=(a+b)/2;
            if(qel_nieves_lindhard_edge(mid,q0,m,s.fermi_energy1,epsilon)>0) b=mid; else a=mid;
        }
        high=qel_min(high,b);
    }
    low=qel_max(low,(qel_max(m,s.fermi_energy2-q0)-s.fermi_energy1)/epsilon);
    return high>low;
}
GPU_HOST_DEVICE inline double qel_nieves_integrand(const QelParameters& p,const QelSample& s,double c,double phi) {
    QelVector l,n; qel_final_state(s,c,phi,l,n);
    if(n.spatial2()<s.kf*s.kf) return 0.;
    QelVector q=p.probe-l;
    double Q2=-q.dot(q), q0=n.t-s.on_shell.t;
    if(Q2<s.q2min || Q2>s.q2max || (s.bound && q0<=0)) return 0.;
    double local=l.t-(p.antineutrino?-1.:1.)*s.coulomb;
    if(local<=p.lepton_mass || l.t-p.lepton_mass<=fabs(s.coulomb)) return 0.;
    double pl=sqrt(l.spatial2()), plocal=sqrt(local*local-p.lepton_mass*p.lepton_mass);
    double factor=plocal*local/(pl*l.t), ratio=plocal/pl;
    QelVector qt{p.probe.x-ratio*l.x,p.probe.y-ratio*l.y,p.probe.z-ratio*l.z,q0};
    double dq=sqrt(qt.spatial2()), q2=q0*q0-dq*dq;
    if(-q2<=1e-6) return 0.;
    double M=(p.initial_mass+p.final_mass)/2,M2=M*M;
    if(s.bound) {
        double a=(-q0+dq*sqrt(1-4*M2/q2))/2;
        double eps=qel_max(M,qel_max(s.fermi_energy2-q0,a));
        double imU=-M2/(6.28318530717958647692*dq)*(s.fermi_energy1-eps);
        if(imU>1e-6) return 0.;
    }
    QelVector k=qel_rotate_to_z(p.probe,qt),t=qel_rotate_to_z(s.on_shell,qt);
    l=qel_rotate_to_z(l,qt);
    double F1,F2,FA,FP; qel_form_factors(p,-q2,F1,F2,FA,FP);
    F1*=.5; F2*=.5; FA=-FA; FP=-FP/M;
    double F12=F1*F1,F22=F2*F2,FA2=FA*FA;
    double pseudoscalar=2*FP*FP*q2+8*FA*FP*M;
    double mixed=16*F1*F2;
    double A00=16*F12*(2*t.t*t.t+2*q0*t.t+q2/2)
        +2*q2*F22*(4-4*t.t*t.t/M2-4*q0*t.t/M2-q0*q0*(4/q2+1/M2))
        +4*FA2*(2*t.t*t.t+2*q0*t.t+q2/2-2*M2)-pseudoscalar*q0*q0-mixed*(-q2+q0*q0);
    double A03=16*F12*(2*t.t*t.z+t.t*dq+t.z*q0)
        +2*q2*F22*(-4*t.t*t.z/M2-2*(dq*t.t+q0*t.z)/M2-dq*q0*(4/q2+1/M2))
        +4*FA2*(2*t.t*t.z+dq*t.t+q0*t.z)-pseudoscalar*dq*q0-mixed*dq*q0;
    double A33=16*F12*(2*t.z*t.z+2*dq*t.z-q2/2)
        +2*q2*F22*(-4-4*t.z*t.z/M2-4*dq*t.z/M2-dq*dq*(4/q2+1/M2))
        +4*FA2*(2*t.z*t.z+2*dq*t.z-q2/2+2*M2)-pseudoscalar*dq*dq-mixed*(q2+dq*dq);
    double A11=16*F12*(2*t.x*t.x-q2/2)+2*q2*F22*(-4-4*t.x*t.x/M2)
        +4*FA2*(2*t.x*t.x-q2/2+2*M2)-mixed*q2;
    double A22=16*F12*(2*t.y*t.y-q2/2)+2*q2*F22*(-4-4*t.y*t.y/M2)
        +4*FA2*(2*t.y*t.y-q2/2+2*M2)-mixed*q2;
    double A12=(p.antineutrino?-1.:1.)*16*FA*(F2+F1)*(-dq*t.t+q0*t.z);
    double kl=k.dot(l);
    double sum=(2*l.t*k.t-kl)*A00-2*(l.t*k.z+l.z*k.t)*A03
        +(2*l.x*k.x+kl)*A11+2*(l.z*k.t-l.t*k.z)*A12
        +(2*l.y*k.y+kl)*A22+(2*l.z*k.z+kl)*A33;
    return p.normalization*.25*factor*sum*s.momentum/(s.root_s*s.on_shell.t*p.probe.t);
}
#endif
