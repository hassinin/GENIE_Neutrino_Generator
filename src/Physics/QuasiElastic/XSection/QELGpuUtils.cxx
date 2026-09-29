#include "Physics/QuasiElastic/XSection/QELGpuUtils.h"
#include "Physics/QuasiElastic/XSection/LwlynSmithQELCCPXSec.h"
#include "Framework/Conventions/Constants.h"
#include "Framework/Conventions/RefFrame.h"
#include "Framework/ParticleData/PDGUtils.h"
#include "Physics/NuclearState/PauliBlocker.h"
#include <cmath>
#include <memory>
#include <typeinfo>

namespace {
gpu_spline::QelVector pack(const TLorentzVector& v) { return {v.X(),v.Y(),v.Z(),v.E()}; }
gpu_spline::QelVector pack(const TVector3& v) { return {v.X(),v.Y(),v.Z(),0}; }
}
bool genie::utils::PrepareQelGpuParameters(const XSecAlgorithmI* model,const Interaction& in,
                                           gpu_spline::QelParameters& p,std::string& reason) {
    p={}; reason.clear();
    if(typeid(*model)!=typeid(LwlynSmithQELCCPXSec) || !in.ProcInfo().IsWeakCC() ||
       !in.ProcInfo().IsQuasiElastic() || in.ExclTag().IsCharmEvent() || in.ExclTag().IsStrangeEvent()) {
        reason="requires Llewellyn-Smith CCQE"; return false;
    }
    const auto* ff=model->SubAlg("FormFactorsAlg");
    if(ff->Id().Name()!="genie::LwlynSmithFFCC" || ff->GetConfig().GetBool("UseElFFTransverseEnhancement")) {
        reason="unsupported QEL form-factor model or transverse enhancement"; return false;
    }
    const auto* elastic=ff->SubAlg("ElasticFormFactorsModel");
    const auto* axial=ff->SubAlg("AxialFormFactorModel");
    if(elastic->Id().Name()!="genie::BBA07ELFormFactorsModel" || axial->Id().Name()!="genie::DipoleAxialFormFactorModel") {
        reason="requires BBA07 elastic and dipole axial form factors"; return false;
    }
    std::unique_ptr<TLorentzVector> probe(in.InitState().GetProbeP4(kRfLab));
    if(!(probe->E()>0) || std::abs(probe->M2())>1e-12*probe->E()*probe->E()) {
        reason="requires a massless incident neutrino"; return false;
    }
    p.probe=pack(*probe);
    p.initial_mass=in.InitState().Tgt().HitNucMass();
    p.final_mass=in.RecoilNucleon()->Mass(); p.lepton_mass=in.FSPrimLepton()->Mass();
    p.proton_mass=constants::kProtonMass; p.neutron_mass=constants::kNeutronMass; p.pion_mass=constants::kPionMass;
    p.antineutrino=pdg::IsAntiNeutrino(in.InitState().ProbePdg());
    const auto& cfg=model->GetConfig();
    double cabibbo=std::cos(cfg.GetDouble("CabibboAngle"));
    const auto& tgt=in.InitState().Tgt();
    int count=pdg::IsProton(tgt.HitNucPdg())?tgt.Z():tgt.N();
    p.normalization=constants::kGF2*cabibbo*cabibbo*cfg.GetDouble("QEL-CC-XSecScale")*count/(8*constants::kPi2);
    p.ma2=std::pow(axial->GetConfig().GetDouble("QEL-Ma"),2);
    p.fa0=axial->GetConfig().GetDouble("QEL-FA0");
    const auto& e=elastic->GetConfig();
    p.mu_p=e.GetDouble("AnomMagnMoment-P"); p.mu_n=e.GetDouble("AnomMagnMoment-N");
    const char* suffix[]{"a1","b1","b2","b3"};
    for(int i=0;i<4;++i) {
        p.gep_rational[i]=e.GetDouble(std::string("BBA07-Gep-")+suffix[i]);
        p.gmp_rational[i]=e.GetDouble(std::string("BBA07-Gmp-")+suffix[i]);
    }
    for(int i=0;i<7;++i) {
        auto s=std::to_string(i+1);
        p.gep_nodes[i]=e.GetDouble("BBA07-Gep-p"+s); p.gmp_nodes[i]=e.GetDouble("BBA07-Gmp-p"+s);
        p.gen_nodes[i]=e.GetDouble("BBA07-Gen-p"+s); p.gmn_nodes[i]=e.GetDouble("BBA07-Gmn-p"+s);
    }
    if(!(p.ma2>0) || p.mu_p==0 || !std::isfinite(p.normalization) || p.normalization<0) {
        reason="unsupported form-factor or normalization parameters"; return false;
    }
    return true;
}
bool genie::utils::PrepareQelGpuSample(const XSecAlgorithmI* model,const Interaction& in,
                                      const gpu_spline::QelParameters& p,gpu_spline::QelSample& s) {
    s={};
    const auto& tgt=in.InitState().Tgt();
    const TLorentzVector initial=tgt.HitNucP4();
    // Spacelike bound nucleons are permitted: the combined system, rather
    // than the struck nucleon alone, must admit a physical COM frame.
    TLorentzVector k(p.probe.x,p.probe.y,p.probe.z,p.probe.t), total=k+initial;
    if(!std::isfinite(total.M2()) || !std::isfinite(total.E())) return false;
    if(!(total.E()>0) || !(total.M2()>0)) return true;
    s.initial=pack(initial);
    TLorentzVector on_shell(initial.Vect(),std::sqrt(p.initial_mass*p.initial_mass+initial.Vect().Mag2()));
    s.on_shell=pack(on_shell);
    s.root_s=std::sqrt(total.M2());
    if(s.root_s<=p.final_mass+p.lepton_mass) return true;
    if(!in.TestBit(kISkipKinematicChk) && !in.PhaseSpace().IsAboveThreshold()) return true;
    auto limits=in.PhaseSpace().Q2Lim();
    if(limits.max<=limits.min) return true;
    s.q2min=limits.min; s.q2max=limits.max;
    s.lepton_com=(total.M2()-p.final_mass*p.final_mass+p.lepton_mass*p.lepton_mass)/(2*s.root_s);
    double p2=s.lepton_com*s.lepton_com-p.lepton_mass*p.lepton_mass;
    if(!(p2>0)) return true;
    s.momentum=std::sqrt(p2); s.nucleon_com=std::sqrt(p2+p.final_mass*p.final_mass);
    TVector3 boost=total.BoostVector();
    s.beta=boost.Mag(); s.gamma=total.E()/s.root_s;
    TVector3 axis=s.beta>1e-15?boost.Unit():k.Vect().Unit();
    TVector3 transverse=k.Vect()-k.Vect().Dot(axis)*axis;
    if(transverse.Mag()<1e-12*k.E()) transverse=axis.Orthogonal();
    transverse=transverse.Unit();
    s.axis=pack(axis); s.transverse=pack(transverse); s.normal=pack(axis.Cross(transverse));
    s.bound=tgt.IsNucleus() && !in.TestBit(kIAssumeFreeNucleon);
    bool pauli=model->GetConfig().GetBool("DoPauliBlocking");
    if(s.bound && pauli) {
        const auto* blocker=dynamic_cast<const PauliBlocker*>(model->SubAlg("PauliBlockerAlg"));
        if(!blocker) return false;
        s.kf=blocker->GetFermiMomentum(tgt,in.RecoilNucleonPdg(),tgt.HitNucPosition());
    }
    s.cos_max=1;
    double bp=s.beta*s.momentum;
    if(bp>0) {
        if(s.bound) s.cos_max=std::min(s.cos_max,((k.E()-on_shell.E()+initial.E())/s.gamma-s.lepton_com)/bp);
        if(s.kf>0) s.cos_max=std::min(s.cos_max,(s.nucleon_com-std::sqrt(p.final_mass*p.final_mass+s.kf*s.kf)/s.gamma)/bp);
    } else {
        if((s.bound && k.E()-s.lepton_com-on_shell.E()+initial.E()<=0) || s.momentum<s.kf) return true;
    }
    s.valid=s.cos_max>-1;
    return true;
}
