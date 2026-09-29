//____________________________________________________________________________
/*
 Copyright (c) 2003-2025, The GENIE Collaboration
 For the full text of the license visit http://copyright.genie-mc.org

*/
//____________________________________________________________________________

#include <TMath.h>
#include <Math/IFunction.h>
#include <Math/IntegratorMultiDim.h>
#include "Math/AdaptiveIntegratorMultiDim.h"

#include "Framework/Algorithm/AlgConfigPool.h"
#include "Framework/Conventions/GBuild.h"
#include "Framework/Conventions/Constants.h"
#include "Framework/Conventions/Controls.h"
#include "Framework/Conventions/Units.h"
#include "Framework/Conventions/KineVar.h"
#include "Framework/Conventions/KinePhaseSpace.h"
#include "Framework/Interaction/Interaction.h"
#include "Framework/Messenger/Messenger.h"
#include "Physics/Multinucleon/XSection/MECXSec.h"
#include "Physics/Multinucleon/XSection/MECUtils.h"
#include "Framework/Numerical/Spline.h"
#include "Framework/ParticleData/PDGCodes.h"
#include "Framework/ParticleData/PDGUtils.h"
#include "Framework/ParticleData/PDGLibrary.h"
#include "Framework/Numerical/MathUtils.h"
#include "Framework/Utils/KineUtils.h"
#include "Framework/Utils/Range1.h"
#include "Framework/Numerical/GSLUtils.h"
#include "Framework/Utils/XSecSplineList.h"
#include "Physics/Multinucleon/XSection/SuSAv2MECPXSec.h"
#include "Physics/Multinucleon/XSection/NievesSimoVacasMECPXSec2016.h"
#include "Physics/QuasiElastic/XSection/SuSAv2QELPXSec.h"
#include "Framework/EventGen/HybridXSecAlgorithm.h"
#include "Physics/HadronTensors/TabulatedLabFrameHadronTensor.h"
#include "Physics/HadronTensors/HadronTensorI.h"
#include "gpu_spline/gpu_hadron_tensor.h"

using namespace genie;
using namespace genie::constants;
using namespace genie::controls;
using namespace genie::utils;

//____________________________________________________________________________
MECXSec::MECXSec() :
XSecIntegratorI("genie::MECXSec")
{

}
//____________________________________________________________________________
MECXSec::MECXSec(string config) :
XSecIntegratorI("genie::MECXSec", config)
{

}
//____________________________________________________________________________
MECXSec::~MECXSec()
{

}
//____________________________________________________________________________
double MECXSec::Integrate(
      const XSecAlgorithmI * model, const Interaction * in) const
{
  if(! model->ValidProcess(in) ) return 0.;

  const KPhaseSpace & kps = in->PhaseSpace(); // only OK phase space for this
  if(!kps.IsAboveThreshold()) {
     LOG("MECXSec", pDEBUG)  << "*** Below energy threshold";
     return 0;
  }

  Interaction interaction(*in);
  interaction.SetBit(kISkipProcessChk);
  interaction.SetBit(kISkipKinematicChk);

  // T, costh limits
  double Enu = in->InitState().ProbeE(kRfLab);
  double LepMass = in->FSPrimLepton()->Mass();
  double TMax = Enu - LepMass;
  double TMin = 0.0;
  double CosthMax = 1.0;
  double CosthMin = -1.0;
  if (Enu < fQ3Max) {
    TMin = 0 ;
    CosthMin = -1 ;
  } else {
    TMin = TMath::Sqrt(TMath::Power(LepMass, 2) + TMath::Power((Enu - fQ3Max), 2)) - LepMass;
    CosthMin = TMath::Sqrt(1 - TMath::Power((fQ3Max / Enu ), 2));
  }

  double kine_min[2] = { TMin,  CosthMin };
  double kine_max[2] = { TMax,  CosthMax };

  // Attempt GPU-accelerated 2D numerical quadrature if GPU is enabled
  if ( gpuspline::GpuHadronTensor::IsGpuEnabled() ) {
    double Q2min = in->ProcInfo().IsEM() ?
      genie::utils::kinematics::electromagnetic::kMinQ2Limit :
      genie::controls::kMinQ2Limit;

    // 1. SuSAv2 MEC
    const SuSAv2MECPXSec* susa_mec = dynamic_cast<const SuSAv2MECPXSec*>(model);
    if ( susa_mec && susa_mec->SupportsGpuTensor() && susa_mec->HadronTensorModel() ) {
      int probe_pdg = in->InitState().ProbePdg();
      int tensor_pdg = kPdgTgtC12;
      HadronTensorType_t tensor_type = (pdg::IsNeutrino(probe_pdg) || pdg::IsAntiNeutrino(probe_pdg)) ?
        kHT_MEC_FullAll : kHT_MEC_EM;
      const TabulatedLabFrameHadronTensor* tensor = dynamic_cast<const TabulatedLabFrameHadronTensor*>(
        susa_mec->HadronTensorModel()->GetTensor(tensor_pdg, tensor_type));
      if ( tensor && tensor->HasGpuTensor() ) {
        double Delta_Q_value = susa_mec->Qvalue(*in);
        double Vud = 0.97427;
        const genie::Registry* temp_reg = genie::AlgConfigPool::Instance()->CommonList("Param", "CKM");
        if (temp_reg) Vud = temp_reg->GetDouble("CKM-Vud");
        double gpu_xsec = 0.0;
        bool ok = tensor->IntegrateGPU(
          probe_pdg, Enu, in->InitState().Probe()->Mass(), LepMass,
          Delta_Q_value, TMin, TMax, CosthMin, CosthMax,
          fQ3Max, Q2min, true /* use_rosenbluth */, Vud,
          gpu_xsec, fGSLRelTol, fGSLMaxEval);
        if ( ok ) {
          double scale = susa_mec->ScalingFactor(*in);
          return gpu_xsec * scale;
        }
      }
    }

    // 2. SuSAv2 QEL 1p1h (direct or wrapped in HybridXSecAlgorithm)
    const SuSAv2QELPXSec* susa_qel = dynamic_cast<const SuSAv2QELPXSec*>(model);
    if ( !susa_qel ) {
      const HybridXSecAlgorithm* hybrid = dynamic_cast<const HybridXSecAlgorithm*>(model);
      if ( hybrid ) {
        susa_qel = dynamic_cast<const SuSAv2QELPXSec*>(hybrid->ChooseXSecAlg(*in));
      }
    }
    if ( susa_qel && susa_qel->IsSuSAv2() && susa_qel->HadronTensorModel() ) {
      int tensor_pdg = kPdgTgtC12;
      HadronTensorType_t tensor_type = kHT_QE_Full;
      if ( in->ProcInfo().IsEM() ) {
        int hit_nuc_pdg = in->InitState().Tgt().HitNucPdg();
        if ( pdg::IsProton(hit_nuc_pdg) ) tensor_type = kHT_QE_EM_proton;
        else if ( pdg::IsNeutron(hit_nuc_pdg) ) tensor_type = kHT_QE_EM_neutron;
        else tensor_type = kHT_QE_EM;
      }
      const TabulatedLabFrameHadronTensor* tensor = dynamic_cast<const TabulatedLabFrameHadronTensor*>(
        susa_qel->HadronTensorModel()->GetTensor(tensor_pdg, tensor_type));
      if ( tensor && tensor->HasGpuTensor() ) {
        double Delta_Q_value = susa_qel->Qvalue(*in);
        double Vud = 0.97427;
        const genie::Registry* temp_reg = genie::AlgConfigPool::Instance()->CommonList("Param", "CKM");
        if (temp_reg) Vud = temp_reg->GetDouble("CKM-Vud");
        double gpu_xsec = 0.0;
        bool ok = tensor->IntegrateGPU(
          in->InitState().ProbePdg(), Enu, in->InitState().Probe()->Mass(), LepMass,
          Delta_Q_value, TMin, TMax, CosthMin, CosthMax,
          fQ3Max, Q2min, true /* use_rosenbluth */, Vud,
          gpu_xsec, fGSLRelTol, fGSLMaxEval);
        if ( ok ) {
          double scale = susa_qel->ScalingFactor(*in);
          return gpu_xsec * scale;
        }
      }
    }

    // 3. Nieves MEC
    const NievesSimoVacasMECPXSec2016* nieves_mec = dynamic_cast<const NievesSimoVacasMECPXSec2016*>(model);
    if ( nieves_mec && nieves_mec->SupportsGpuTensor() && nieves_mec->HadronTensorModel() ) {
      int target_pdg = in->InitState().Tgt().Pdg();
      int probe_pdg = in->InitState().ProbePdg();
      int tensor_pdg = kPdgTgtC12;
      if ( target_pdg == tensor_pdg ) {
        const bool pn = in->InitState().Tgt().HitNucPdg() == kPdgClusterNP;
        const bool delta = in->ExclTag().KnownResonance();
        HadronTensorType_t tensor_type = delta ?
          (pn ? kHT_MEC_Deltapn : kHT_MEC_DeltaAll) :
          (pn ? kHT_MEC_Fullpn : kHT_MEC_FullAll);
        const TabulatedLabFrameHadronTensor* tensor = dynamic_cast<const TabulatedLabFrameHadronTensor*>(
          nieves_mec->HadronTensorModel()->GetTensor(tensor_pdg, tensor_type));
        if ( tensor && tensor->HasGpuTensor() ) {
          double Q_value = genie::utils::mec::Qvalue(target_pdg, probe_pdg);
          double gpu_xsec = 0.0;
          bool ok = tensor->IntegrateGPU(
            probe_pdg, Enu, in->InitState().Probe()->Mass(), LepMass,
            Q_value, TMin, TMax, CosthMin, CosthMax,
            fQ3Max, Q2min, false /* use_rosenbluth */, 0.97427,
            gpu_xsec, fGSLRelTol, fGSLMaxEval);
          if ( ok ) {
            return gpu_xsec * nieves_mec->TensorScale(*in);
          }
        }
      }
    }
  }

  double xsec = 0;

  double abstol = 1; //We mostly care about relative tolerance.
  genie::utils::mec::gsl::d2Xsec_dTCosth func(model, interaction, Enu, LepMass );
  ROOT::Math::IntegrationMultiDim::Type ig_type =
    utils::gsl::IntegrationNDimTypeFromString(fGSLIntgType);
  ROOT::Math::IntegratorMultiDim ig(func, ig_type, abstol, fGSLRelTol, fGSLMaxEval);

  xsec = ig.Integral(kine_min, kine_max);

  return xsec;
}
//____________________________________________________________________________
void MECXSec::Configure(const Registry & config)
{
  Algorithm::Configure(config);
  this->LoadConfig();
}
//____________________________________________________________________________
void MECXSec::Configure(string config)
{
  Algorithm::Configure(config);
  this->LoadConfig();
}
//____________________________________________________________________________
void MECXSec::LoadConfig(void)
{
  GetParam( "NSV-Q3Max", fQ3Max ) ;

  // Get GSL integration type & relative tolerance
  GetParamDef( "gsl-integration-type", fGSLIntgType, string("vegas") ) ;

  int max ;
  GetParamDef( "gsl-max-eval", max, 20000 ) ;
  fGSLMaxEval    = (unsigned int) max ;

  GetParamDef( "gsl-relative-tolerance", fGSLRelTol, 0.01 ) ;
  GetParamDef( "split-integral", fSplitIntegral, true ) ;

}
