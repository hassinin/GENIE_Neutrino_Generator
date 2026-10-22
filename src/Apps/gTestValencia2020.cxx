// Regression checks for the Valencia2020 table interface and paper sampler.
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <TRandom3.h>
#include <TFile.h>
#include <TTree.h>
#include "Framework/Ntuple/NtpMCEventRecord.h"
#include "Framework/GHEP/GHepParticle.h"
#include "Physics/NuclearState/NuclearModelI.h"
#include <unistd.h>
#include "Framework/Algorithm/AlgFactory.h"
#include "Framework/Conventions/Constants.h"
#include "Framework/Conventions/Units.h"
#include "Framework/Interaction/Interaction.h"
#include "Framework/ParticleData/PDGCodes.h"
#include "Framework/ParticleData/PDGLibrary.h"
#include "Physics/HadronTensors/Valencia2020HadronTensorModel.h"
#include "Physics/HadronTensors/LabFrameHadronTensorI.h"
#include "Physics/Multinucleon/XSection/Valencia2020MECPXSec.h"
#include "Physics/Multinucleon/XSection/MECUtils.h"
#include "Physics/Multinucleon/EventGen/Valencia2020Kinematics.h"
#include "Physics/Multinucleon/EventGen/Valencia2020NuWro.h"

using namespace genie;
void Require(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error(message);
}
void Near(double a, double b, double tolerance, const std::string& message) {
  Require(std::isfinite(a) && std::abs(a-b) <= tolerance*std::max(std::abs(b),1.e-30), message);
}
void ValidateEvents(const char* path, const char* expected_channel, bool nuwro=false) {
  const auto nuclear = dynamic_cast<const NuclearModelI*>(AlgFactory::Instance()->GetAlgorithm("genie::LocalFGM","Default"));
  Require(nuclear,"Cannot load local Fermi gas for validation");
  TFile file(path,"READ");
  Require(!file.IsZombie(),"Cannot open event file");
  auto tree = dynamic_cast<TTree*>(file.Get("gtree"));
  Require(tree && tree->GetEntries()>0,"Missing/empty event tree");
  NtpMCEventRecord* record=nullptr;
  tree->SetBranchAddress("gmcrec",&record);
  int n2=0,n3=0; double max_closure=0.,max_final_closure=0.,max_shell=0.,min_pauli=1.e9;
  double min_excitation=1.e9,sum_excitation=0.;
  int unpolarized=0;
  const std::string channel(expected_channel);
  for (Long64_t ie=0;ie<tree->GetEntries();++ie) {
    tree->GetEntry(ie);
    auto event=record->event;
    const auto hit=event->HitNucleon();
    Require(hit,"Missing hit cluster after ROOT round-trip");
    const int n=pdg::NucleonClusterSize(hit->Pdg());
    Require(n==2 || n==3,"Invalid initial cluster");
    const int hitpos=event->HitNucleonPosition();
    GHepParticle* recoil=nullptr; GHepParticle* remnant=nullptr;
    int recoilpos=-1;
    for (int j=0;j<event->GetEntriesFast();++j) {
      auto particle=event->Particle(j);
      if (particle->FirstMother()==hitpos && pdg::IsNucleonCluster(particle->Pdg())) {
        recoil=particle; recoilpos=j;
      }
      if (particle->FirstMother()==event->TargetNucleusPosition() &&
          pdg::IsIon(particle->Pdg())) remnant=particle;
    }
    Require(recoil && remnant,"Missing recoil/remnant ancestry");
    std::vector<GHepParticle*> daughters;
    for (int j=0;j<event->GetEntriesFast();++j) {
      auto particle=event->Particle(j);
      if (particle->FirstMother()==recoilpos && pdg::IsNucleon(particle->Pdg())) daughters.push_back(particle);
    }
    Require(int(daughters.size())==n,"Primary nucleon multiplicity mismatch");
    Require(channel!="3p3h" || n==3,"3p3h generated a two-body state");
    Require(channel=="all" || channel=="legacy" || channel=="3p3h" || n==2,"2p2h generated a three-body state");
    n2+=(n==2); n3+=(n==3);
    Target target(event->TargetNucleus()->Pdg());
    const double radius=event->Probe()->X4()->Vect().Mag();
    TLorentzVector sum;
    int protons=0;
    for (auto particle:daughters) {
      protons+=(particle->Pdg()==kPdgProton);
      const double mass=PDGLibrary::Instance()->Find(particle->Pdg())->Mass();
      const double shell=std::abs(particle->P4()->M2()-mass*mass);
      max_shell=std::max(max_shell,shell);
      Require(shell<1.e-8,"Off-shell primary nucleon");
      if (channel!="legacy" && !nuwro) {
        const double kf=nuclear->LocalFermiMomentum(target,particle->Pdg(),radius);
        const double margin=particle->P4()->P()-kf;
        min_pauli=std::min(min_pauli,margin);
        Require(margin > -1.e-10,"Primary nucleon violates local Pauli blocking");
      }
      sum+=*particle->P4();
    }
    if (nuwro && n==2) {
      const auto& density=utils::valencia2020::NuWroDensityFor(target.Z(),target.N());
      const int first_initial=hit->Pdg()==kPdgClusterPP ? kPdgProton:kPdgNeutron;
      const double kf=density.FermiMomentum(first_initial,radius);
      const double mean_mass=.5*(PDGLibrary::Instance()->Find(kPdgProton)->Mass()
                               +PDGLibrary::Instance()->Find(kPdgNeutron)->Mass());
      const double kinetic_fermi=std::sqrt(mean_mass*mean_mass+kf*kf)-mean_mass;
      const double margin=daughters[1]->P4()->E()
        -PDGLibrary::Instance()->Find(daughters[1]->Pdg())->Mass()-kinetic_fermi;
      min_pauli=std::min(min_pauli,margin);
      Require(margin > -1.e-10,"NuWro backward kinetic Pauli cut violated");
      TLorentzVector first=*daughters[0]->P4();
      const TVector3 boost=recoil->P4()->BoostVector(); first.Boost(-boost);
      Require(first.Vect().Dot(boost)>-1.e-12,"NuWro forward role reversed");
    }
    if (channel!="legacy") {
      // CC V-A helicity: lepton polarised against, antilepton along its momentum.
      // Event files from older builds may carry no polarization: count them.
      auto lepton=event->FinalStatePrimaryLepton();
      if (!lepton->PolzIsSet()) ++unpolarized;
      else {
        TVector3 polz; lepton->GetPolarization(polz);
        const double helicity=polz.Unit().Dot(lepton->P4()->Vect().Unit());
        Near(helicity,lepton->Pdg()>0 ? -1.:1.,1.e-9,"Wrong primary lepton helicity");
      }
    }
    {
      const double ground=PDGLibrary::Instance()->Find(remnant->Pdg())->Mass();
      const double excitation=remnant->P4()->M()-ground;
      min_excitation=std::min(min_excitation,excitation);
      sum_excitation+=excitation;
    }
    Require(remnant->A()+n==target.A(),"Baryon number not conserved");
    const int lepton_charge=event->Probe()->Pdg()>0 ? -1:1;
    Require(remnant->Z()+protons+lepton_charge==target.Z(),"Charge not conserved");
    if (channel!="legacy") Require(event->Summary()->ExclTag().NNucleons()==n,"Exclusive count tag mismatch");
    const TLorentzVector closure=sum+*remnant->P4()+*event->FinalStatePrimaryLepton()->P4()
      -*event->TargetNucleus()->P4()-*event->Probe()->P4();
    const double error=std::max(std::abs(closure.E()),closure.Vect().Mag());
    max_closure=std::max(max_closure,error);
    Require(error<1.e-8,"Pre-FSI event four-momentum not conserved");
    TLorentzVector final_sum;
    for (int j=0;j<event->GetEntriesFast();++j) {
      const auto particle=event->Particle(j);
      if (particle->Status()==kIStStableFinalState ||
          particle->Status()==kIStFinalStateNuclearRemnant) final_sum+=*particle->P4();
    }
    const TLorentzVector final_error=final_sum-*event->TargetNucleus()->P4()-*event->Probe()->P4();
    const double final_delta=std::max(std::abs(final_error.E()),final_error.Vect().Mag());
    max_final_closure=std::max(max_final_closure,final_delta);
    Require(final_delta<1.e-8,"Post-FSI event four-momentum not conserved");
    const TLorentzVector cluster_error=sum-*recoil->P4();
    Require(std::max(std::abs(cluster_error.E()),cluster_error.Vect().Mag())<1.e-8,"Cluster decay not conserved");
    if (channel=="np" || channel=="pn") {
      const int forward=(channel=="np") ? kPdgNeutron:kPdgProton;
      const int expected=event->Probe()->Pdg()>0 ? forward:pdg::SwitchProtonNeutron(forward);
      Require(daughters[0]->Pdg()==expected,"Selected mixed-pair channel lost");
      TLorentzVector first=*daughters[0]->P4();
      const TVector3 boost=recoil->P4()->BoostVector();
      first.Boost(-boost);
      Require(first.Vect().Dot(boost)>-1.e-12,"Wrong forward species");
    }
    record->Clear();
  }
  std::cout << "PASS events " << path << ": two=" << n2 << " three=" << n3
    << " max_dP=" << max_closure << " GeV max_final_dP=" << max_final_closure
    << " GeV max_mass2_error=" << max_shell
    << (nuwro ? " min_NuWro_backward_kinetic_margin=" : " min_local_Pauli_margin=")
    << min_pauli << " GeV prescription=" << (nuwro ? "NuWroCompatible":"GENIE")
    << " min_remnant_excitation=" << min_excitation
    << " GeV mean_remnant_excitation=" << sum_excitation/std::max(1,n2+n3) << " GeV"
    << " unpolarized_leptons=" << unpolarized << std::endl;
}

int main(int argc, char** argv) {
 try {
  if (argc>2 && std::string(argv[1])=="--events") {
    const std::string prescription=argc>4 ? argv[4]:"GENIE";
    Require(prescription=="GENIE" || prescription=="NuWroCompatible","Unknown event validation prescription");
    ValidateEvents(argv[2],argc>3 ? argv[3]:"all",prescription=="NuWroCompatible");
    _exit(0);
  }
  const auto factory = AlgFactory::Instance();
  const auto tensors = dynamic_cast<const Valencia2020HadronTensorModel*>(
    factory->GetAlgorithm("genie::Valencia2020HadronTensorModel","Default"));
  // Optional third positional argument selects a named cross-section config.
  const std::string config = argc>3 ? argv[3] : "Default";
  const auto model = dynamic_cast<const Valencia2020MECPXSec*>(
    factory->GetAlgorithm("genie::Valencia2020MECPXSec",config));
  Require(tensors && model, "Algorithm loading failed");
  const auto nuwro_shift = dynamic_cast<const Valencia2020MECPXSec*>(
    factory->GetAlgorithm("genie::Valencia2020MECPXSec","NuWroEnergyShift"));
  Require(nuwro_shift,"NuWro energy-shift configuration failed");
  for (int target : {kPdgTgtC12,kPdgTgtO16,kPdgTgtCa40,1000180400,
                     1000060130,1000070140,1000120240,1000130270}) {
    const int Z=pdg::IonPdgCodeToZ(target);
    const double expected_nu=Z>12 ? .013809 : (Z>6 ? .014906 : .016827);
    const double expected_anti=Z>12 ? .001822 : (Z>6 ? .010931 : .013878);
    for (int probe : {14,-14,12,-12}) {
      std::unique_ptr<Interaction> in(Interaction::MECCC(target,probe,1.));
      Near(nuwro_shift->EnergyShift(in.get()),probe>0 ? expected_nu : expected_anti,
           1.e-14,"Incorrect NuWro energy-shift constant or Z boundary");
    }
  }
  using utils::valencia2020::ProjectChannelWeights;
  // A negative partial must lower the inclusive rate, without introducing
  // negative sampling probabilities or moving strength into/out of 3p3h.
  const auto projected = ProjectChannelWeights({{4.,-1.,2.,3.}});
  Near(projected[0],10./3.,1.e-14,"Positive-channel ratio changed");
  Near(projected[1],0.,1.e-14,"Negative channel sampled");
  Near(projected[2],5./3.,1.e-14,"Positive-channel ratio changed");
  Near(projected[3],3.,1.e-14,"3p3h rate altered by 2p2h projection");
  const auto cancelled = ProjectChannelWeights({{1.,-4.,2.,3.}});
  Require(cancelled[0]+cancelled[1]+cancelled[2]==0. && cancelled[3]==3.,
          "Negative 2p2h total borrowed 3p3h strength");
  const auto positive = ProjectChannelWeights({{4.,1.,2.,3.}});
  Require(positive==std::array<double,4>{{4.,1.,2.,3.}},"Positive channels changed");
  const auto negative = ProjectChannelWeights({{-1.,-2.,-3.,-4.}});
  Require(negative==std::array<double,4>{{0.,0.,0.,0.}},"Negative sector rate survived");
  using utils::valencia2020::NuWroChannelWeights;
  Require(NuWroChannelWeights({{4.,-1.,2.,3.}})==std::array<double,4>{{4.,0.,1.,3.}},
          "NuWro conditional mixed-channel sampling changed");
  Require(NuWroChannelWeights({{4.,-3.,1.,3.}})==std::array<double,4>{{2.,0.,0.,3.}},
          "NuWro same-charge probability above one changed the sector rate");
  Require(NuWroChannelWeights({{1.,-4.,2.,3.}})==std::array<double,4>{{0.,0.,0.,3.}},
          "NuWro negative sector survived");
  TRandom3 weight_rng(271828);
  for (int trial=0; trial<100000; ++trial) {
    std::array<double,4> raw;
    for (double& value:raw) value=(2.*weight_rng.Rndm()-1.)*1.e-10;
    const auto weights=NuWroChannelWeights(raw);
    for (double value:weights) Require(value>=0.,"Channel-weight roundoff became negative");
    Near(weights[0]+weights[1]+weights[2],std::max(0.,raw[0]+raw[1]+raw[2]),
         1.e-14,"NuWro projection changed signed sector strength");
  }
  const HadronTensorType_t types[] = {kHT_MEC_Valencia2020_pp,kHT_MEC_Valencia2020_np,
    kHT_MEC_Valencia2020_pn,kHT_MEC_Valencia2020_3p3h};
  for (int target : {kPdgTgtC12,kPdgTgtO16,kPdgTgtCa40})
   for (auto type : types) {
    const auto tensor = dynamic_cast<const LabFrameHadronTensorI*>(tensors->GetTensor(target,type));
    Require(tensor, "Missing tensor");
    Require(tensor->q0Min() == 0., "Missing zero-transfer boundary");
    Near(tensor->qMagMax(),1.2,1.e-12,"Wrong table extent");
    Require(std::abs(tensor->tt(0.,.5)) == 0.,"Response at zero energy transfer");
    Near(tensor->tt(.005,.5).real(),.5*tensor->tt(.01,.5).real(),1.e-10,"Low-transfer interpolation");
   }
  const auto carbon = dynamic_cast<const LabFrameHadronTensorI*>(tensors->GetTensor(kPdgTgtC12,types[0]));
  Near(carbon->tt(.01,.01).real(),.00387965,1.e-10,"MeV^-1 to GeV^-1 conversion missing");
  // Independent raw-MeV fixture. At cos(theta)=-.5 the native NuWro and GENIE
  // massive-lepton angular conventions coincide, isolating the unit conversion.
  double reference = 0.;
  for (auto type : types) {
    const auto t = dynamic_cast<const LabFrameHadronTensorI*>(tensors->GetTensor(kPdgTgtC12,type));
    reference += 12.*t->dSigma_dT_dCosTheta(14,.8,0.,.3-.105658369,-.5,.105658369,.016827)/units::cm2;
  }
  Near(reference,4.2243802534939e-39,1.e-6,"Independent absolute-normalization fixture");

  // Proxy table: in the GENIE prescription every A >= 22 (Ar40, Fe56, Pb208)
  // must use the Ca40 per-nucleon tensors scaled by its own A, not C12.
  {
    const auto genie_model = dynamic_cast<const Valencia2020MECPXSec*>(
      factory->GetAlgorithm("genie::Valencia2020MECPXSec","Default"));
    Require(genie_model,"Default configuration failed to load");
    for (int target : {1000180400, 1000260560, 1000822080, 1000060120}) {
      std::unique_ptr<Interaction> in(Interaction::MECCC(target,14,1.));
      in->KinePtr()->SetKV(kKVTl,.6); in->KinePtr()->SetKV(kKVctl,.8);
      double raw[5];
      genie_model->GetRawChannelCrossSections(in.get(),raw[0],raw[1],raw[2],raw[3],raw[4]);
      const int proxy = pdg::IonPdgCodeToA(target) >= 22 ? kPdgTgtCa40 : kPdgTgtC12;
      const double shift = genie_model->EnergyShift(in.get());
      for (int i=0; i<4; ++i) {
        const auto t = dynamic_cast<const LabFrameHadronTensorI*>(tensors->GetTensor(proxy,types[i]));
        Near(raw[i],pdg::IonPdgCodeToA(target)*t->dSigma_dT_dCosTheta(in.get(),shift),1.e-12,
             "Wrong proxy tensor table for target "+std::to_string(target));
      }
      Require(raw[4] > 0.,"No MEC response for target "+std::to_string(target));
    }
    std::cout << "PASS proxy tables: Ar40, Fe56, Pb208 -> Ca40; C12 -> C12\n";
  }

  // W4 massive-lepton term. NuWroCompatible must use the exact contraction
  // (sin^2(theta/2)); only NuWroNativeW4 reproduces NuWro's sin^2(theta) typo.
  // Antineutrino, forward muon: a point where the two forms differ by ~3x.
  {
    const auto compatible = dynamic_cast<const Valencia2020MECPXSec*>(
      factory->GetAlgorithm("genie::Valencia2020MECPXSec","NuWroCompatible"));
    const auto native = dynamic_cast<const Valencia2020MECPXSec*>(
      factory->GetAlgorithm("genie::Valencia2020MECPXSec","NuWroNativeW4"));
    Require(compatible && native,"NuWro W4 configurations failed to load");
    Require(!compatible->NuWroW4SinTheta() && native->NuWroW4SinTheta(),
            "Wrong NuWroW4SinTheta settings");
    std::unique_ptr<Interaction> in(Interaction::MECCC(kPdgTgtC12,-14,.5));
    in->KinePtr()->SetKV(kKVTl,.2); in->KinePtr()->SetKV(kKVctl,.8);
    double exact[5], typo[5];
    compatible->GetRawChannelCrossSections(in.get(),exact[0],exact[1],exact[2],exact[3],exact[4]);
    native->GetRawChannelCrossSections(in.get(),typo[0],typo[1],typo[2],typo[3],typo[4]);
    const double shift = compatible->EnergyShift(in.get());
    for (int i=0; i<4; ++i) {
      const auto t = dynamic_cast<const LabFrameHadronTensorI*>(tensors->GetTensor(kPdgTgtC12,types[i]));
      Near(exact[i],12.*t->dSigma_dT_dCosTheta(in.get(),shift),1.e-12,
           "NuWroCompatible W4 term is not the exact lepton-tensor contraction");
    }
    Require(exact[4] > 0. && std::abs(typo[4]-exact[4]) > 1.e-6*exact[4],
            "NuWroNativeW4 switch has no effect");
    std::cout << "PASS W4 contraction: NuWroNativeW4/exact = " << typo[4]/exact[4] << "\n";
  }

  std::ofstream points;
  if (argc > 1) {
    points.open(argv[1]);
    Require(points.good(),"Cannot open reference-point CSV");
    points << "target,probe,E,T,cos,Qvalue,pp,np,pn,three,total,raw_pp,raw_np,raw_pn,raw_three,raw_total\n" << std::setprecision(17);
  }
  int negative_partial_points=0;
  for (int target : {kPdgTgtC12,kPdgTgtO16,kPdgTgtCa40,1000180400})
   for (int probe : {14,-14,12,-12})
    for (double E : {.5,1.,2.})
     for (double fraction : {.25,.55,.8})
      for (double cosine : {-.5,.5,.9,.99}) {
       std::unique_ptr<Interaction> in(Interaction::MECCC(target,probe,E));
       if (config=="Default")
         Near(model->EnergyShift(in.get()),utils::mec::Qvalue(target,probe),1.e-14,
              "Default target Q-value changed");
       double mass = in->FSPrimLepton()->Mass();
       const double T = (E-mass)*fraction;
       in->KinePtr()->SetKV(kKVTl,T); in->KinePtr()->SetKV(kKVctl,cosine);
       double pp,np,pn,three,total;
       model->GetChannelCrossSections(in.get(),pp,np,pn,three,total);
       double rp,rn,rr,r3,rt;
       model->GetRawChannelCrossSections(in.get(),rp,rn,rr,r3,rt);
       Near(total,std::max(0.,rp+rn+rr)+std::max(0.,r3),1.e-12,
            "Inclusive signed response was changed by channel projection");
       Require(pp>=0. && np>=0. && pn>=0. && three>=0.,"Negative sampling weight");
       if (rp<0. || rn<0. || rr<0.) {
         ++negative_partial_points;
         Require(total < std::max(0.,rp)+std::max(0.,rn)+std::max(0.,rr)+std::max(0.,r3),
                 "Regression: negative partial clipped before inclusive sum");
       }
       Near(model->XSec(in.get(),kPSTlctl),total,1.e-12,"Channel sum mismatch");
       Require(total >= 0. && std::isfinite(total),"Invalid cross section");
       if (points) points << target << ',' << probe << ',' << E << ',' << T << ',' << cosine << ','
         << model->EnergyShift(in.get()) << ',' << pp/units::cm2 << ',' << np/units::cm2 << ','
         << pn/units::cm2 << ',' << three/units::cm2 << ',' << total/units::cm2 << ','
         << rp/units::cm2 << ',' << rn/units::cm2 << ',' << rr/units::cm2 << ','
         << r3/units::cm2 << ',' << rt/units::cm2 << '\n';
       in->InitStatePtr()->TgtPtr()->SetHitNucPdg(kPdgClusterNP);
       Near(model->XSec(in.get(),kPSTlctl),pp,1.e-12,"Initial NP channel mismatch");
       in->InitStatePtr()->TgtPtr()->SetHitNucPdg(probe>0 ? kPdgClusterNN:kPdgClusterPP);
       Near(model->XSec(in.get(),kPSTlctl),np+pn,1.e-12,"Charge-conjugate channel mismatch");
      }
  if (points) points.close();
  Require(negative_partial_points>0,"No real negative-channel regression points exercised");
  std::cout << "PASS signed inclusive rate and positive projection: "
            << negative_partial_points << " negative-partial points\n";
  if (argc>2) {
    std::ofstream integrals(argv[2]);
    Require(integrals.good(),"Cannot open integral CSV");
    integrals << "target,probe,E,Qvalue,sigma\n" << std::setprecision(17);
    for (int target:{kPdgTgtC12,kPdgTgtO16,kPdgTgtCa40,1000180400})
     for (int probe:{14,-14})
      for (double E:{.5,1.,2.}) {
        std::unique_ptr<Interaction> in(Interaction::MECCC(target,probe,E));
        const double sigma=model->Integral(in.get())/units::cm2;
        Require(sigma>0. && std::isfinite(sigma),"Invalid integral");
        integrals << target << ',' << probe << ',' << E << ','
          << model->EnergyShift(in.get()) << ',' << sigma << '\n';
      }
  }
  // The corrected-coordinate guard must suppress negative energy transfers.
  std::unique_ptr<Interaction> boundary(Interaction::MECCC(kPdgTgtC12,14,1.));
  boundary->KinePtr()->SetKV(kKVTl,1.-boundary->FSPrimLepton()->Mass()-.001);
  boundary->KinePtr()->SetKV(kKVctl,.99);
  Require(model->XSec(boundary.get(),kPSTlctl) == 0.,"Negative corrected transfer is nonzero");

  using namespace utils::valencia2020;
  TRandom3 rng(24111523);
  for (auto fit : {std::make_pair(.77,4),std::make_pair(.7,3),std::make_pair(.8,4),std::make_pair(-.5,3)})
   for (double kappa : {1.,.3}) {
    const double P = fit.first; const int l = fit.second;
    const double constant = P >= 0. ? 1.-P : 1.;
    const double norm = constant + P/(l+1.);
    const double expected = (constant/2.+P/(l+2.))/norm;
    const double second = (constant/3.+P/(l+3.))/norm;
    double sum=0.; int below=0; const int n=100000;
    for (int i=0;i<n;++i) {
      const double c=SampleForwardCosine(rng,P,l,0.,kappa)/kappa;
      Require(c>=0. && c<=1.,"Angular sample outside support");
      sum+=c; below+=(c<.5);
    }
    Require(std::abs(sum/n-expected)<6.*std::sqrt((second-expected*expected)/n),"Angular moment disagrees with analytic law");
    const double cdf=(constant*.5+P*std::pow(.5,l+1)/(l+1))/norm;
    Require(std::abs(double(below)/n-cdf)<6.*std::sqrt(cdf*(1.-cdf)/n),"Angular CDF disagrees with analytic law");
   }
  double lower, upper;
  const double mp=PDGLibrary::Instance()->Find(kPdgProton)->Mass();
  const double W=2.05, momentum=.6, energy=std::sqrt(W*W+momentum*momentum);
  Require(PauliAngularRange(momentum/energy,std::sqrt(W*W/4.-mp*mp),W/2.,W/2.,
    std::sqrt(mp*mp+.221*.221),std::sqrt(mp*mp+.221*.221),lower,upper),"Expected nonempty Pauli interval");
  Require(upper<.87 && upper>.85,"Pauli blocking counterexample was not excluded");
  Require(!PauliAngularRange(0.,.1,1.,1.,1.1,1.1,lower,upper),"Zero-boost blocked state accepted");
  Near(ThreeBodyChargeWeight(6,6,3,false),20.,1.e-12,"Three-body combinatorics");
  Near(ThreeBodyChargeWeight(6,6,2,false),90.,1.e-12,"Three-body combinatorics");
  Require(ThreeBodyChargeWeight(6,6,0,false)==0.,"Neutrino charge violation");
  Near(NuWroThreeBodyChargeWeight(6,6,1,false),70.,1.e-12,"NuWro shifted carbon pnn weight");
  Near(NuWroThreeBodyChargeWeight(6,6,2,false),105.,1.e-12,"NuWro shifted carbon ppn weight");
  Near(NuWroThreeBodyChargeWeight(6,6,3,false),35.,1.e-12,"NuWro shifted carbon ppp weight");
  for (int z=0;z<=3;++z)
    Near(NuWroThreeBodyChargeWeight(6,6,z,false),NuWroThreeBodyChargeWeight(6,6,3-z,true),
         1.e-12,"NuWro carbon charge conjugation");
  Require(NuWroAngularChannel(0,true)==1 && NuWroAngularChannel(1,true)==2 &&
          NuWroAngularChannel(2,true)==1,"NuWro antineutrino angular assignment");
  for (int z=0;z<=3;++z) {
    const int code=kPdgClusterNNN+z;
    Require(!pdg::Is2NucleonCluster(code) && pdg::Is3NucleonCluster(code),"Cluster classification");
    Near(PDGLibrary::Instance()->Find(code)->Charge(),3.*z,1.e-12,"Cluster charge");
  }
  // Without a channel-isolation overlay the unmasked tables are the tables.
  {
    std::unique_ptr<Interaction> in(Interaction::MECCC(kPdgTgtC12,14,1.));
    in->KinePtr()->SetKV(kKVTl,.6); in->KinePtr()->SetKV(kKVctl,.8);
    double a[5], b[5];
    model->GetRawChannelCrossSections(in.get(),a[0],a[1],a[2],a[3],a[4]);
    model->GetUnmaskedRawChannelCrossSections(in.get(),b[0],b[1],b[2],b[3],b[4]);
    for (int i=0;i<5;++i) Near(b[i],a[i],1.e-14,"Unmasked tables differ without an overlay");
    Require(a[4] > 0.,"No response at the unmasked-table check point");
  }
  // 3-nucleon clusters are registered in code, not in the catalogue file:
  // they must survive a database reload. Done last: reload resets the table.
  PDGLibrary::Instance()->ReloadDBase();
  for (int z=0; z<=3; ++z)
    Require(PDGLibrary::Instance()->Find(kPdgClusterNNN+z) != nullptr,
            "3-nucleon cluster lost after PDGLibrary::ReloadDBase");
  std::cout << "PASS unmasked tables; 3N clusters survive PDG reload\n";
  std::cout << std::setprecision(12) << "Reference d2sigma = " << reference << " cm^2/GeV\n"
    << "PASS: units, grids, 576 reference points, channels, angular statistics, Pauli bounds, clusters\n" << std::flush;
  // Avoid GENIE's process-global algorithm-pool shutdown ordering.
  _exit(0);
 } catch (const std::exception& error) {
  std::cerr << "FAIL: " << error.what() << std::endl;
  _exit(1);
 }
}
