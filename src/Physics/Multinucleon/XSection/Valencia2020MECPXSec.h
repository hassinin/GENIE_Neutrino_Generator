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

  // One setting controls the contraction, exclusive weights and hadronizer.
  bool NuWroCompatible() const { return fPreFSIPrescription == "NuWro"; }

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

  // Override Algorithm::Configure
  void Configure (const Registry & config);
  void Configure (string config);

private:

  void LoadConfig (void);

  double fXSecCCScale;
  double fXSecNCScale;
  bool   fInclude3p3h;
  string fEnergyShiftPrescription;
  string fPreFSIPrescription;

  const HadronTensorModelI * fHadronTensorModel;
  const XSecIntegratorI *    fXSecIntegrator;
  const XSecScaleI *         fMECScaleAlg;
  const QvalueShifter *      fQvalueShifter;
};

}       // genie namespace

#endif  // _VALENCIA2020_MEC_PXSEC_H_
