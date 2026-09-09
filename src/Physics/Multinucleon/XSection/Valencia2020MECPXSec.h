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

  // Channel-decomposed cross section calculation
  void GetChannelCrossSections(
    const Interaction * i,
    double & xsec_pp,
    double & xsec_np,
    double & xsec_pn,
    double & xsec_3p3h,
    double & xsec_tot) const;

  // Override Algorithm::Configure
  void Configure (const Registry & config);
  void Configure (string config);

private:

  void LoadConfig (void);

  double fXSecCCScale;
  double fXSecNCScale;
  bool   fInclude3p3h;

  const HadronTensorModelI * fHadronTensorModel;
  const XSecIntegratorI *    fXSecIntegrator;
  const XSecScaleI *         fMECScaleAlg;
  const QvalueShifter *      fQvalueShifter;
};

}       // genie namespace

#endif  // _VALENCIA2020_MEC_PXSEC_H_
