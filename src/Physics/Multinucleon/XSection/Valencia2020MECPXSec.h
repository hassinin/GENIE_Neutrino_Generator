//____________________________________________________________________________
/*!

\class    genie::Valencia2020MECPXSec

\brief    Computes the Valencia 2020 exclusive MEC differential cross section
          with channel decomposition (pp, np, pn, 3p3h).
          Reference: J.E. Sobczyk, J. Nieves, F. Sanchez, PRC 102, 024601 (2020).

\cpright  Copyright (c) 2003-2026, The GENIE Collaboration
          For the full text of the license visit http://copyright.genie-mc.org
*/
//____________________________________________________________________________

#ifndef _VALENCIA2020_MEC_PXSEC_H_
#define _VALENCIA2020_MEC_PXSEC_H_

#include <set>
#include <utility>

#include "Framework/EventGen/XSecAlgorithmI.h"
#include "Physics/HadronTensors/HadronTensorModelI.h"
#include "Physics/Common/XSecScaleI.h"
#include "Physics/Common/QvalueShifter.h"
#include "Physics/Multinucleon/XSection/Valencia2020ChannelWeights.h"

namespace genie {

class XSecIntegratorI;

class Valencia2020MECPXSec : public XSecAlgorithmI {

public:
  Valencia2020MECPXSec();
  Valencia2020MECPXSec(string config);
  virtual ~Valencia2020MECPXSec();

  // XSecAlgorithmI interface implementation
  double XSec         (const Interaction * i, KinePhaseSpace_t k) const;
  double Integral     (const Interaction * i) const;
  bool   ValidProcess (const Interaction * i) const;

  // Tensor energy shift in GeV, including an optional QvalueShifterAlg.
  // TargetQValue (default) or the installed NuWro Valencia2020 prescription.
  double EnergyShift(const Interaction * i) const;

  // One setting controls the exclusive weights and hadronizer.
  bool NuWroCompatible() const { return fPreFSIPrescription == "NuWro"; }

  // Off by default: every configuration uses the exact L_mu_nu W^mu_nu
  // contraction, whose massive-lepton W4 term carries sin^2(theta/2).
  // On: reproduce native NuWro's sin^2(theta) typo (mecevent_2020Valencia.cc,
  // also mecevent_Nieves.cc and mecevent_SuSA.cc), for validation only.
  bool NuWroW4SinTheta() const { return fNuWroW4SinTheta; }

  // Nonnegative effective channel cross sections, GeV^-3, including scales.
  // Default rescales positive partials; NuWro uses conditional channel draws.
  // Their sum plus the nonnegative 3p3h rate equals the inclusive cross section.
  void GetChannelCrossSections(
    const Interaction * i,
    double & xsec_pp,
    double & xsec_np,
    double & xsec_pn,
    double & xsec_3p3h,
    double & xsec_tot) const;

  // Signed contractions for diagnostics; NOT probabilities. Total is the raw
  // signed sum, before the positivity constraint on each multiplicity sector.
  void GetRawChannelCrossSections(
    const Interaction * i, double & pp, double & np, double & pn,
    double & three, double & total) const;

  // Same, but always with the full (unmasked) Valencia tables, even when a
  // channel-isolation overlay of Valencia2020HadronTensorModel.xml zeroes
  // some channels.
  // Used for the NuWro-compatible 3p3h Pauli flag, which in native NuWro
  // depends on the physical 2p2h response.
  void GetUnmaskedRawChannelCrossSections(
    const Interaction * i, double & pp, double & np, double & pn,
    double & three, double & total) const;

  // Override Algorithm::Configure
  void Configure (const Registry & config);
  void Configure (string config);

private:

  void LoadConfig (void);
  void RawChannelCrossSections(const HadronTensorModelI * model,
    const Interaction * i, double & pp, double & np, double & pn,
    double & three, double & total) const;

  // TargetQValue prescription: utils::mec::Qvalue needs the target and its
  // CC daughter (Z+1 for nu, Z-1 for nubar) in the PDG table and exit()s
  // otherwise. Checked once per (target, probe sign); cached below.
  bool TargetQValueAvailable(int target_pdg, int probe_pdg) const;

  double fXSecCCScale;
  double fXSecNCScale;   // UNUSED: read for configuration compatibility only;
                         // Valencia2020 is CC-only (ValidProcess requires CC)
  bool   fInclude3p3h;
  bool   fNuWroW4SinTheta;
  string fEnergyShiftPrescription;
  string fPreFSIPrescription;

  const HadronTensorModelI * fHadronTensorModel;
  const HadronTensorModelI * fUnmaskedHadronTensorModel; // == fHadronTensorModel unless configured
  const XSecIntegratorI *    fXSecIntegrator;
  const XSecScaleI *         fMECScaleAlg;
  const QvalueShifter *      fQvalueShifter;

  // Per-target one-time bookkeeping (diagnostics only, no physics state).
  // Keys: (target PDG, +1 nu / -1 nubar) for the Q-value daughter lookup.
  mutable std::set<std::pair<int,int> > fQvalueTargetsOK;
  mutable std::set<std::pair<int,int> > fQvalueTargetsMissing; // warned once
  mutable std::set<int> fLightTargetsNotified; // A<9 -> C12 proxy notice
};

}       // genie namespace

#endif  // _VALENCIA2020_MEC_PXSEC_H_
