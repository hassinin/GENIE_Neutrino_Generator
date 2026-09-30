//____________________________________________________________________________
/*
 Copyright (c) 2003-2025, The GENIE Collaboration
 For the full text of the license visit http://copyright.genie-mc.org

 Steven Gardiner <gardiner \at fnal.gov>
 Fermi National Accelerator Laboratory
*/
//____________________________________________________________________________

#include "Framework/Algorithm/AlgConfigPool.h"
#include "Framework/Algorithm/AlgFactory.h"
#include "Framework/Conventions/GBuild.h"
#include "Framework/Conventions/Constants.h"
#include "Framework/Conventions/Controls.h"
#include "Framework/Conventions/Units.h"
#include "Framework/Conventions/KineVar.h"
#include "Framework/Conventions/RefFrame.h"
#include "Physics/QuasiElastic/XSection/NewQELXSec.h"

#include "Physics/NuclearState/NuclearModelI.h"
#include "Physics/XSectionIntegration/GSLXSecFunc.h"
#include "Framework/Messenger/Messenger.h"
#include "Framework/Numerical/RandomGen.h"
#include "Framework/ParticleData/PDGUtils.h"
#include "Framework/Utils/KineUtils.h"
#include "Physics/NuclearState/NuclearUtils.h"
#include "Framework/Utils/Range1.h"
#include "Framework/Numerical/GSLUtils.h"
#include "Physics/Common/VertexGenerator.h"
#include "Physics/NuclearState/NuclearModel.h"
#include "Physics/NuclearState/NuclearModelMap.h"
#ifdef __GENIE_GPU_ENABLED__
#include "Physics/QuasiElastic/XSection/QELGpuUtils.h"
#include "gpu_spline/gpu_qel_integrator.h"
#include "gpu_spline/gpu_hadron_tensor.h"
#include "gpu_spline/gpu_status.h"
#endif
#include <vector>
#include <cmath>

using namespace genie;
using namespace genie::constants;
using namespace genie::utils::gsl;

//____________________________________________________________________________
NewQELXSec::NewQELXSec() : XSecIntegratorI("genie::NewQELXSec")
{

}
//____________________________________________________________________________
NewQELXSec::NewQELXSec(std::string config) : XSecIntegratorI("genie::NewQELXSec", config)
{

}
//____________________________________________________________________________
double NewQELXSec::Integrate(const XSecAlgorithmI* model, const Interaction* in) const
{
  LOG("NewQELXSec",pDEBUG) << "Beginning integrate";
  if ( !model->ValidProcess(in) ) return 0.;

  Interaction* interaction = new Interaction( *in );
  interaction->SetBit( kISkipProcessChk );
  //interaction->SetBit( kISkipKinematicChk );

  const NuclearModelI* nucl_model = dynamic_cast<const NuclearModelI*>(
    model->SubAlg("IntegralNuclearModel") );
  assert( nucl_model );

  AlgFactory* algf = AlgFactory::Instance();
  const VertexGenerator* vtx_gen = dynamic_cast<const VertexGenerator*>(
    algf->GetAlgorithm(fVertexGenID) );
  assert( vtx_gen );

  // Determine the appropriate binding energy mode to use.
  // The default given here is for the case of a free nucleon.
  QELEvGen_BindingMode_t bind_mode = kOnShell;
  Target* tgt = interaction->InitState().TgtPtr();
  if ( tgt->IsNucleus() ) {
    std::string bind_mode_str = model->GetConfig()
      .GetString( "IntegralNucleonBindingMode" );
    bind_mode = genie::utils::StringToQELBindingMode( bind_mode_str );
  }

  utils::gsl::FullQELdXSec* func = new utils::gsl::FullQELdXSec(model,
    interaction, bind_mode, fMinAngleEM);
  ROOT::Math::IntegrationMultiDim::Type ig_type =
    utils::gsl::IntegrationNDimTypeFromString( fGSLIntgType );

  // Switch to using the copy of the interaction in the integrator rather than
  // the copy that we made in this function
  delete interaction;
  interaction = func->GetInteractionPtr();

  // Also update the pointer to the Target
  tgt = interaction->InitState().TgtPtr();

  double abstol = 1e-16; // We mostly care about relative tolerance
  ROOT::Math::IntegratorMultiDim ig(*func, ig_type, abstol, fGSLRelTol, fGSLMaxEval);

  // Integration ranges for the lepton COM frame scattering angles (in the
  // kPSQELEvGen phase space, these are measured with respect to the COM
  // velocity as observed in the lab frame)
  Range1D_t cos_theta_0_lim( -1., 1. );
  Range1D_t phi_0_lim( 0., 2.*kPi );

  double kine_min[2] = { cos_theta_0_lim.min, phi_0_lim.min };
  double kine_max[2] = { cos_theta_0_lim.max, phi_0_lim.max };

  // If averaging over the initial nucleon distribution has been
  // disabled, just integrate over angles and return the result.
  if ( !fAverageOverNucleons ) {
    double xsec_total = ig.Integral(kine_min, kine_max);
    delete func;
    return xsec_total;
  }

  // For a free nucleon target (hit nucleon is at rest in the lab frame), we
  // don't need to do an MC integration over the initial state variables. In
  // this case, just set up the nucleon at the origin, on-shell, and at rest,
  // then integrate over the angles and return the result.

  double probeE = interaction->InitState().ProbeE( kRfLab );
  if ( !tgt->IsNucleus() ) {
    tgt->SetHitNucPosition(0.);

    if ( tgt->IsNucleus() ) nucl_model->GenerateNucleon(*tgt, 0.);
    else {
      nucl_model->SetRemovalEnergy(0.);
      interaction->SetBit( kIAssumeFreeNucleon );
    }

    nucl_model->SetMomentum3( TVector3(0., 0., 0.) );
    double xsec_total = ig.Integral(kine_min, kine_max);
    delete func;
    return xsec_total;
  }

  // For a nuclear target, we need to loop over a bunch of nucleons sampled
  // from the nuclear model (with positions sampled from the vertex generator
  // to allow for using the local Fermi gas model). The MC estimator for the
  // total cross section is simply the mean of ig.Integral() for all of the
  // sampled nucleons.
#ifdef __GENIE_GPU_ENABLED__
  gpu_spline::QelParameters gpu_parameters{};
  std::string gpu_reason;
  bool use_gpu = fUseGpuIntegration && gpuspline::GpuHadronTensor::IsGpuEnabled()
    && fGSLIntgType == "adaptive" && fNumNucleonThrows > 0
    && utils::PrepareQelGpuParameters(model, *interaction, gpu_parameters, gpu_reason);
  if (fUseGpuIntegration && gpuspline::GpuHadronTensor::IsGpuEnabled() && !use_gpu) {
    LOG("NewQELXSec", pNOTICE) << "[GPU CCQE] CPU fallback: "
      << (gpu_reason.empty() ? "unsupported integration settings" : gpu_reason);
  }
  struct NuclearSample { TVector3 momentum; double removal, radius; bool prepared; };
  std::vector<NuclearSample> nuclear_samples;
  std::vector<gpu_spline::QelSample> gpu_samples;
  if (use_gpu) { nuclear_samples.reserve(fNumNucleonThrows); gpu_samples.reserve(fNumNucleonThrows); }
#endif
  double xsec_sum = 0.;
  for (int n = 0; n < fNumNucleonThrows; ++n) {

    // Select a new position for the initial hit nucleon (needed for the local
    // Fermi gas model, but other than slowing things down a bit, it doesn't
    // hurt to do this for other models)
    TVector3 vertex_pos = vtx_gen->GenerateVertex( interaction, tgt->A() );
    double radius = vertex_pos.Mag();
    tgt->SetHitNucPosition( radius );

    // Sample a new nucleon 3-momentum and removal energy (this will be applied
    // to the nucleon via a call to genie::utils::ComputeFullQELPXSec(), so
    // there's no need to mess with its 4-momentum here)
    nucl_model->GenerateNucleon(*tgt, radius);

    // The initial state variables have all been defined, so integrate over
    // the final lepton angles.
#ifdef __GENIE_GPU_ENABLED__
    if (use_gpu) {
      double binding_energy=0.;
      utils::BindHitNucleon(*interaction, *nucl_model, binding_energy, bind_mode);
      gpu_spline::QelSample sample{};
      bool prepared=utils::PrepareQelGpuSample(model, *interaction, gpu_parameters, sample);
      nuclear_samples.push_back({nucl_model->Momentum3(),nucl_model->RemovalEnergy(),radius,prepared});
      gpu_samples.push_back(sample);
    } else
#endif
    {
      double xsec = ig.Integral(kine_min, kine_max);
      xsec_sum += xsec;
    }
  }

#ifdef __GENIE_GPU_ENABLED__
  if (use_gpu) {
    std::vector<gpu_spline::QelIntegralResult> values;
    bool device_error=false;
    bool ok=gpu_spline::IntegrateQelGpu(gpu_parameters, gpu_samples, fGSLRelTol,
      fGSLMaxEval, gpuspline::GpuHadronTensor::DefaultDevice(), values, gpu_reason, &device_error);
    int adaptive_recovered=0;
    unsigned long long adaptive_evaluations=0;
    std::string adaptive_error;
    if (ok && fUseGpuAdaptiveIntegration) {
      const auto original=values;
      if (gpu_spline::RefineQelGpu(gpu_parameters, gpu_samples, fGSLRelTol,
          fGpuAdaptiveMaxEval, gpuspline::GpuHadronTensor::DefaultDevice(), values, adaptive_error,
          &device_error)) {
        for (size_t n=0;n<values.size();++n) {
          adaptive_recovered+=!original[n].converged && values[n].converged;
          adaptive_evaluations+=values[n].evaluations-original[n].evaluations;
        }
      }
      // RefineQelGpu preserves the first-stage results on a device error.
      // Only still-unconverged samples need the existing CPU fallback.
    }
    // A CUDA error (not an invalid request or numerical problem) disables the
    // GPU for the rest of the job. This knot finishes through the per-sample
    // CPU fallback below; later knots see IsGpuEnabled() == false.
    if (device_error && gpu_spline::DisableGpu(ok ? adaptive_error : gpu_reason)) {
      LOG("NewQELXSec", pERROR) << "[GPU CCQE] CUDA error: " << (ok ? adaptive_error : gpu_reason)
        << ". Disabling the GPU for the rest of this job; continuing on the CPU.";
    }
    int cpu_fallbacks=0, gpu_integrals=0, empty_domains=0;
    for (int n=0;n<fNumNucleonThrows;++n) {
      if (ok && nuclear_samples[n].prepared && values[n].converged) {
        xsec_sum+=values[n].value;
        if (gpu_samples[n].valid) ++gpu_integrals;
        else ++empty_domains;
      } else {
        // Restore this exact sample. Redrawing after a device/convergence
        // failure would bias the nuclear average and change the RNG stream.
        const auto& sample=nuclear_samples[n];
        tgt->SetHitNucPosition(sample.radius);
        nucl_model->SetMomentum3(sample.momentum);
        nucl_model->SetRemovalEnergy(sample.removal);
        xsec_sum+=ig.Integral(kine_min,kine_max);
        ++cpu_fallbacks;
      }
    }
    // Preserve the nuclear model's externally visible final sampled state.
    const auto& last=nuclear_samples.back();
    tgt->SetHitNucPosition(last.radius);
    nucl_model->SetMomentum3(last.momentum);
    nucl_model->SetRemovalEnergy(last.removal);
    LOG("NewQELXSec", pNOTICE) << "[GPU CCQE] E=" << probeE << ": "
      << gpu_integrals << " GPU integrals, " << empty_domains << " empty domains, "
      << cpu_fallbacks << " CPU fallbacks / " << fNumNucleonThrows << " nuclear samples"
      << (ok ? "" : "; device error: " + gpu_reason);
    if (fUseGpuAdaptiveIntegration) {
      LOG("NewQELXSec", pNOTICE) << "[GPU CCQE adaptive] E=" << probeE << ": "
        << adaptive_recovered << " recovered integrals, " << adaptive_evaluations
        << " extra evaluations; per-sample budget=" << fGpuAdaptiveMaxEval
        << (adaptive_error.empty() ? "" : "; device error: " + adaptive_error);
    }
  }
#endif

  delete func;

  // MC estimator of the total cross section is the mean of the xsec values
  double xsec_mean = xsec_sum / fNumNucleonThrows;

  return xsec_mean;
}
//____________________________________________________________________________
void NewQELXSec::Configure(const Registry & config)
{
  Algorithm::Configure(config);
  this->LoadConfig();
}
//____________________________________________________________________________
void NewQELXSec::Configure(string config)
{
  Algorithm::Configure(config);
  this->LoadConfig();
}
//____________________________________________________________________________
void NewQELXSec::LoadConfig(void)
{
  // Get GSL integration type & relative tolerance
  GetParamDef( "gsl-integration-type", fGSLIntgType, std::string("adaptive") ) ;
  GetParamDef( "gsl-relative-tolerance", fGSLRelTol, 1e-2 ) ;
  int max;
  GetParamDef( "gsl-max-eval", max, 500000 ) ;
  fGSLMaxEval  = static_cast<unsigned int>( max );

  RgAlg vertexGenID;
  GetParamDef( "VertexGenAlg", vertexGenID, RgAlg("genie::VertexGenerator", "Default") );
  fVertexGenID = AlgId( vertexGenID );

  GetParamDef( "NumNucleonThrows", fNumNucleonThrows, 5000 );

  // TODO: This is a parameter that may also be specified in the XML
  // configuration for QELEventGenerator. Avoid duplication here to ensure
  // consistency.
  GetParamDef( "SF-MinAngleEMscattering", fMinAngleEM, 0. ) ;

  // If true, then the integration of the total cross section will include an
  // MC integration over the initial state nuclear model
  GetParamDef( "AverageOverNucleons", fAverageOverNucleons, true );
  GetParamDef( "UseGPUIntegration", fUseGpuIntegration, true );
  GetParamDef( "UseGPUAdaptiveIntegration", fUseGpuAdaptiveIntegration, true );
  int adaptive_max;
  GetParamDef( "GPUAdaptiveMaxEval", adaptive_max, 100000 );
  fGpuAdaptiveMaxEval=adaptive_max>0?static_cast<unsigned>(adaptive_max):0;
}

genie::utils::gsl::FullQELdXSec::FullQELdXSec(const XSecAlgorithmI* xsec_model,
  const Interaction* interaction, QELEvGen_BindingMode_t binding_mode, double min_angle_EM)
  : fXSecModel( xsec_model ), fInteraction( new Interaction(*interaction) ),
  fHitNucleonBindingMode( binding_mode ), fMinAngleEM( min_angle_EM )
{
  fNuclModel = dynamic_cast<const NuclearModelI*>( fXSecModel->SubAlg("IntegralNuclearModel") );
  assert( fNuclModel );
}

genie::utils::gsl::FullQELdXSec::~FullQELdXSec()
{
  delete fInteraction;
}

Interaction* genie::utils::gsl::FullQELdXSec::GetInteractionPtr()
{
  return fInteraction;
}

const Interaction& genie::utils::gsl::FullQELdXSec::GetInteraction() const
{
  return *fInteraction;
}

ROOT::Math::IBaseFunctionMultiDim* genie::utils::gsl::FullQELdXSec::Clone(void) const
{
  return new FullQELdXSec(fXSecModel, fInteraction, fHitNucleonBindingMode, fMinAngleEM);
}

unsigned int genie::utils::gsl::FullQELdXSec::NDim(void) const
{
  return 2;
}

double genie::utils::gsl::FullQELdXSec::DoEval(const double* xin) const
{
  // Elements of "xin"
  //
  // element 0: "cos_theta0" = Cosine of theta0, the angle between the COM frame
  //                           3-momentum of the outgoing lepton and the COM frame velocity
  //                           as measured in the laboratory frame
  // element 1: "phi_theta0" = Azimuthal angle of the COM frame 3-momentum of the
  //                           outgoing lepton measured with respect to the COM frame
  //                           velocity as measured in the laboratory frame

  double cos_theta0 = xin[0];
  double phi0 = xin[1];

  // Dummy storage for the binding energy of the hit nucleon
  double dummy_Eb = 0.;

  // Compute the full differential cross section
  double xsec = genie::utils::ComputeFullQELPXSec(fInteraction, fNuclModel,
    fXSecModel, cos_theta0, phi0, dummy_Eb, fHitNucleonBindingMode, fMinAngleEM, true);

  return xsec;
}
