#include <iostream>
#include <iomanip>
#include <string>
#include "Framework/Algorithm/AlgFactory.h"
#include "Framework/Algorithm/AlgConfigPool.h"
#include "Framework/Interaction/Interaction.h"
#include "Framework/Conventions/KineVar.h"
#include "Framework/Conventions/KinePhaseSpace.h"
#include "Framework/ParticleData/PDGCodes.h"
#include "Framework/Messenger/Messenger.h"
#include "Physics/HadronTensors/Valencia2020HadronTensorModel.h"
#include "Physics/HadronTensors/LabFrameHadronTensorI.h"
#include "Physics/Multinucleon/XSection/Valencia2020MECPXSec.h"

int main() {
  std::cout << "========================================================" << std::endl;
  std::cout << "Testing Valencia 2020 Model Implementation in GENIE" << std::endl;
  std::cout << "========================================================" << std::endl;

  // 1. Test Hadron Tensor Model
  std::cout << "\n--- 1. Testing Valencia2020HadronTensorModel ---" << std::endl;
  genie::AlgFactory* alg_factory = genie::AlgFactory::Instance();
  const genie::Valencia2020HadronTensorModel* tensor_model =
    dynamic_cast<const genie::Valencia2020HadronTensorModel*>(
      alg_factory->GetAlgorithm("genie::Valencia2020HadronTensorModel/Default"));

  if (!tensor_model) {
    std::cerr << "ERROR: Failed to load genie::Valencia2020HadronTensorModel/Default" << std::endl;
    return 1;
  }
  std::cout << "SUCCESS: Valencia2020HadronTensorModel instantiated." << std::endl;

  int test_targets[] = {genie::kPdgTgtC12, genie::kPdgTgtO16, genie::kPdgTgtCa40};
  std::string target_names[] = {"12C", "16O", "40Ca"};

  for (int i = 0; i < 3; ++i) {
    int tgt = test_targets[i];
    const genie::LabFrameHadronTensorI* t_pp =
      dynamic_cast<const genie::LabFrameHadronTensorI*>(
        tensor_model->GetTensor(tgt, genie::kHT_MEC_Valencia2020_pp));
    const genie::LabFrameHadronTensorI* t_np =
      dynamic_cast<const genie::LabFrameHadronTensorI*>(
        tensor_model->GetTensor(tgt, genie::kHT_MEC_Valencia2020_np));
    const genie::LabFrameHadronTensorI* t_pn =
      dynamic_cast<const genie::LabFrameHadronTensorI*>(
        tensor_model->GetTensor(tgt, genie::kHT_MEC_Valencia2020_pn));
    const genie::LabFrameHadronTensorI* t_3p3h =
      dynamic_cast<const genie::LabFrameHadronTensorI*>(
        tensor_model->GetTensor(tgt, genie::kHT_MEC_Valencia2020_3p3h));

    if (t_pp && t_np && t_pn && t_3p3h) {
      std::cout << "Target " << target_names[i] << " (" << tgt << "): ALL 4 TENSORS LOADED (pp, np, pn, 3p3h)." << std::endl;
      std::cout << "  Grid q0: [" << t_pp->q0Min() << ", " << t_pp->q0Max()
                << "] GeV, qMag: [" << t_pp->qMagMin() << ", " << t_pp->qMagMax() << "] GeV" << std::endl;
    } else {
      std::cerr << "ERROR: Missing tensor for target " << target_names[i] << std::endl;
      return 1;
    }
  }

  // 2. Test Cross Section Model
  std::cout << "\n--- 2. Testing Valencia2020MECPXSec ---" << std::endl;
  const genie::Valencia2020MECPXSec* xsec_model =
    dynamic_cast<const genie::Valencia2020MECPXSec*>(
      alg_factory->GetAlgorithm("genie::Valencia2020MECPXSec/Default"));

  if (!xsec_model) {
    std::cerr << "ERROR: Failed to load genie::Valencia2020MECPXSec/Default" << std::endl;
    return 1;
  }
  std::cout << "SUCCESS: Valencia2020MECPXSec instantiated." << std::endl;

  // Test CC nu_mu on 12C and 40Ar
  int targets[] = {1000060120, 1000180400};
  std::string names[] = {"12C (1000060120)", "40Ar (1000180400 - Ca40 proxy)"};

  double Ev = 1.0; // GeV
  double Tl = 0.5; // GeV
  double ctl = 0.8; // cos(theta)

  for (int i = 0; i < 2; ++i) {
    int tgt_pdg = targets[i];
    std::cout << "\nKinematics: Ev = " << Ev << " GeV, Tmu = " << Tl << " GeV, cos(theta) = " << ctl << std::endl;
    std::cout << "Evaluating CC nu_mu on " << names[i] << ":" << std::endl;

    genie::Interaction * interaction = genie::Interaction::MECCC(tgt_pdg, genie::kPdgNuMu, Ev);
    interaction->KinePtr()->SetKV(genie::kKVTl, Tl);
    interaction->KinePtr()->SetKV(genie::kKVctl, ctl);

    double x_pp=0., x_np=0., x_pn=0., x_3p3h=0., x_tot=0.;
    xsec_model->GetChannelCrossSections(interaction, x_pp, x_np, x_pn, x_3p3h, x_tot);

    std::cout << std::scientific << std::setprecision(5);
    std::cout << "  d2sigma/dT_dcos (pp)   = " << x_pp   << " 10^-38 cm^2/GeV" << std::endl;
    std::cout << "  d2sigma/dT_dcos (np)   = " << x_np   << " 10^-38 cm^2/GeV" << std::endl;
    std::cout << "  d2sigma/dT_dcos (pn)   = " << x_pn   << " 10^-38 cm^2/GeV" << std::endl;
    std::cout << "  d2sigma/dT_dcos (3p3h) = " << x_3p3h << " 10^-38 cm^2/GeV" << std::endl;
    std::cout << "  d2sigma/dT_dcos (TOT)  = " << x_tot  << " 10^-38 cm^2/GeV" << std::endl;
    if (x_tot > 0) {
      std::cout << "  Fractions: pp=" << (x_pp/x_tot)*100. << "%, np=" << (x_np/x_tot)*100.
                << "%, pn=" << (x_pn/x_tot)*100. << "%, 3p3h=" << (x_3p3h/x_tot)*100. << "%" << std::endl;
    }

    double diff_xsec = xsec_model->XSec(interaction, genie::kPSTlctl);
    std::cout << "  XSec(kPSTlctl) returned = " << diff_xsec << std::endl;
  }

  std::cout << "\n========================================================" << std::endl;
  std::cout << "All Valencia 2020 tests completed successfully!" << std::endl;
  std::cout << "========================================================" << std::endl;
  _exit(0);
}
