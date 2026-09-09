//_________________________________________________________________________
/*
 Copyright (c) 2003-2026, The GENIE Collaboration
 For the full text of the license visit http://copyright.genie-mc.org

 Valencia 2020 exclusive MEC model implementation.
 Reference: J.E. Sobczyk, J. Nieves, F. Sanchez, PRC 102, 024601 (2020).
*/
//_________________________________________________________________________

#include "Framework/Algorithm/AlgConfigPool.h"
#include "Framework/Conventions/Constants.h"
#include "Framework/Conventions/Units.h"
#include "Framework/Messenger/Messenger.h"
#include "Framework/ParticleData/PDGCodes.h"
#include "Framework/ParticleData/PDGLibrary.h"
#include "Framework/ParticleData/PDGUtils.h"
#include "Framework/Utils/KineUtils.h"
#include "Physics/HadronTensors/Valencia2020HadronTensorModel.h"
#include "Physics/HadronTensors/LabFrameHadronTensorI.h"
#include "Physics/Multinucleon/XSection/Valencia2020MECPXSec.h"
#include "Physics/Multinucleon/XSection/MECUtils.h"
#include "Physics/XSectionIntegration/XSecIntegratorI.h"

using namespace genie;
using namespace genie::constants;

//_________________________________________________________________________
Valencia2020MECPXSec::Valencia2020MECPXSec() :
XSecAlgorithmI("genie::Valencia2020MECPXSec")
{

}
//_________________________________________________________________________
Valencia2020MECPXSec::Valencia2020MECPXSec(string config) :
XSecAlgorithmI("genie::Valencia2020MECPXSec", config)
{

}
//_________________________________________________________________________
Valencia2020MECPXSec::~Valencia2020MECPXSec()
{

}
//_________________________________________________________________________
void Valencia2020MECPXSec::GetChannelCrossSections(
  const Interaction * interaction,
  double & xsec_pp,
  double & xsec_np,
  double & xsec_pn,
  double & xsec_3p3h,
  double & xsec_tot) const
{
  xsec_pp = 0.;
  xsec_np = 0.;
  xsec_pn = 0.;
  xsec_3p3h = 0.;
  xsec_tot = 0.;

  int target_pdg = interaction->InitState().Tgt().Pdg();
  int A_request = pdg::IonPdgCodeToA(target_pdg);

  int tensor_pdg = target_pdg;
  // The raw Valencia 2020 tables are tabulated per nucleon.
  // Scale by the target mass number A_request to obtain the whole-nucleus cross section.
  double a_scale = (double)A_request;

  if (A_request >= 9 && A_request < 15) {
    tensor_pdg = kPdgTgtC12;
  }
  else if (A_request >= 15 && A_request < 22) {
    tensor_pdg = kPdgTgtO16;
  }
  else if (A_request >= 22 && A_request <= 50) {
    // 40Ca grid (also used directly as proxy for 40Ar)
    tensor_pdg = kPdgTgtCa40;
  }
  else {
    tensor_pdg = kPdgTgtC12;
  }

  const LabFrameHadronTensorI* tensor_pp = dynamic_cast<const LabFrameHadronTensorI*>(
    fHadronTensorModel->GetTensor(tensor_pdg, genie::kHT_MEC_Valencia2020_pp));
  const LabFrameHadronTensorI* tensor_np = dynamic_cast<const LabFrameHadronTensorI*>(
    fHadronTensorModel->GetTensor(tensor_pdg, genie::kHT_MEC_Valencia2020_np));
  const LabFrameHadronTensorI* tensor_pn = dynamic_cast<const LabFrameHadronTensorI*>(
    fHadronTensorModel->GetTensor(tensor_pdg, genie::kHT_MEC_Valencia2020_pn));
  const LabFrameHadronTensorI* tensor_3p3h = dynamic_cast<const LabFrameHadronTensorI*>(
    fHadronTensorModel->GetTensor(tensor_pdg, genie::kHT_MEC_Valencia2020_3p3h));

  if (!tensor_pp || !tensor_np || !tensor_pn) {
    LOG("Valencia2020MEC", pWARN) << "Failed to load hadron tensors for pdg " << tensor_pdg;
    return;
  }

  // Check kinematic range
  double Tl = interaction->Kine().GetKV(kKVTl);
  double cos_l = interaction->Kine().GetKV(kKVctl);
  double ml = interaction->FSPrimLepton()->Mass();
  double El = Tl + ml;
  double Ev = interaction->InitState().ProbeE(kRfLab);
  double Q0 = Ev - El;

  double mv = interaction->InitState().Probe()->Mass();
  double pv = std::sqrt(std::max(0., Ev*Ev - mv*mv));
  double pl = std::sqrt(std::max(0., El*El - ml*ml));
  double Q3 = std::sqrt(std::max(0., pv*pv + pl*pl - 2.0*pv*pl*cos_l));

  if (Q0 < tensor_pp->q0Min() || Q0 > tensor_pp->q0Max() ||
      Q3 < tensor_pp->qMagMin() || Q3 > tensor_pp->qMagMax()) {
    return;
  }

  int nu_pdg = interaction->InitState().ProbePdg();
  double Q_value = genie::utils::mec::Qvalue(target_pdg, nu_pdg);
  if (fQvalueShifter) {
    Q_value += Q_value * fQvalueShifter->Shift(interaction->InitState().Tgt());
  }

  xsec_pp = tensor_pp->dSigma_dT_dCosTheta(interaction, Q_value) * a_scale;
  xsec_np = tensor_np->dSigma_dT_dCosTheta(interaction, Q_value) * a_scale;
  xsec_pn = tensor_pn->dSigma_dT_dCosTheta(interaction, Q_value) * a_scale;

  if (fInclude3p3h && tensor_3p3h) {
    xsec_3p3h = tensor_3p3h->dSigma_dT_dCosTheta(interaction, Q_value) * a_scale;
  }

  xsec_tot = xsec_pp + xsec_np + xsec_pn + xsec_3p3h;
}
//_________________________________________________________________________
double Valencia2020MECPXSec::XSec(
  const Interaction * interaction, KinePhaseSpace_t kps) const
{
  if (kps == kPSWQ2fE) {
    double Q2 = interaction->Kine().GetKV(kKVQ2);
    double W = interaction->Kine().GetKV(kKVW);

    const InitialState& init_state = interaction->InitState();
    double mv = init_state.Probe()->Mass();
    double Ev = init_state.ProbeE(kRfLab);
    double pv = std::sqrt(std::max(0., Ev*Ev - mv*mv));

    const TLorentzVector& hit_nuc_P4 = init_state.Tgt().HitNucP4();
    double M = hit_nuc_P4.M();

    double ml = interaction->FSPrimLepton()->Mass();
    double Tl = Ev - ml - ((W*W + Q2 - M*M) / (2.*M));
    double El = Tl + ml;
    double pl = std::sqrt(std::max(0., El*El - ml*ml));
    double ctl = (2.*Ev*El - Q2 - mv*mv - ml*ml) / (2. * pv * pl);

    interaction->KinePtr()->SetKV(kKVTl, Tl);
    interaction->KinePtr()->SetKV(kKVctl, ctl);
  }

  double xsec_pp, xsec_np, xsec_pn, xsec_3p3h, xsec_tot;
  this->GetChannelCrossSections(interaction, xsec_pp, xsec_np, xsec_pn, xsec_3p3h, xsec_tot);

  double xsec = xsec_tot;

  // Specific hit cluster channel query (if requested)
  int hit_nuc = interaction->InitState().Tgt().HitNucPdg();
  if (hit_nuc == kPdgClusterNP) {
    // In neutrino CC, the pp outgoing channel comes from initial np
    xsec = xsec_pp;
  } else if (hit_nuc == kPdgClusterNN) {
    xsec = xsec_np + xsec_pn;
  }

  // Scale CC / NC
  const ProcessInfo& proc_info = interaction->ProcInfo();
  if (proc_info.IsWeakCC()) xsec *= fXSecCCScale;
  else if (proc_info.IsWeakNC()) xsec *= fXSecNCScale;

  if (fMECScaleAlg) xsec *= fMECScaleAlg->GetScaling(*interaction);

  if (kps != kPSTlctl && kps != kPSWQ2fE) {
    LOG("Valencia2020MEC", pWARN)
      << "Unsupported transformation from "
      << KinePhaseSpace::AsString(kPSTlctl) << " to "
      << KinePhaseSpace::AsString(kps);
    xsec = 0;
  } else if (kps == kPSWQ2fE && xsec != 0.) {
    double J = utils::kinematics::Jacobian(interaction, kPSTlctl, kps);
    xsec *= J;
  }

  return xsec;
}
//_________________________________________________________________________
double Valencia2020MECPXSec::Integral(const Interaction * interaction) const
{
  double xsec = fXSecIntegrator->Integrate(this, interaction);
  return xsec;
}
//_________________________________________________________________________
bool Valencia2020MECPXSec::ValidProcess(const Interaction * interaction) const
{
  if (interaction->TestBit(kISkipProcessChk)) return true;
  const ProcessInfo & proc_info = interaction->ProcInfo();
  if (!proc_info.IsMEC()) return false;
  return true;
}
//_________________________________________________________________________
void Valencia2020MECPXSec::Configure(const Registry & config)
{
  Algorithm::Configure(config);
  this->LoadConfig();
}
//____________________________________________________________________________
void Valencia2020MECPXSec::Configure(string config)
{
  Algorithm::Configure(config);
  this->LoadConfig();
}
//_________________________________________________________________________
void Valencia2020MECPXSec::LoadConfig(void)
{
  bool good_config = true;

  GetParamDef("MEC-CC-XSecScale", fXSecCCScale, 1.0);
  GetParamDef("MEC-NC-XSecScale", fXSecNCScale, 1.0);
  GetParamDef("Include3p3h", fInclude3p3h, true);

  fHadronTensorModel = dynamic_cast<const HadronTensorModelI *>(this->SubAlg("HadronTensorAlg"));
  if (!fHadronTensorModel) {
    good_config = false;
    LOG("Valencia2020MECPXSec", pERROR) << "The required HadronTensorAlg does not exist.";
  }

  fXSecIntegrator = dynamic_cast<const XSecIntegratorI *>(this->SubAlg("NumericalIntegrationAlg"));
  if (!fXSecIntegrator) {
    good_config = false;
    LOG("Valencia2020MECPXSec", pERROR) << "The required NumericalIntegrationAlg does not exist.";
  }

  fQvalueShifter = nullptr;
  if (GetConfig().Exists("QvalueShifterAlg")) {
    fQvalueShifter = dynamic_cast<const QvalueShifter *>(this->SubAlg("QvalueShifterAlg"));
  }

  fMECScaleAlg = nullptr;
  if (GetConfig().Exists("MECScaleAlg")) {
    fMECScaleAlg = dynamic_cast<const XSecScaleI *>(this->SubAlg("MECScaleAlg"));
  }

  if (!good_config) {
    LOG("Valencia2020MECPXSec", pFATAL) << "Configuration has failed.";
    exit(78);
  }
}
