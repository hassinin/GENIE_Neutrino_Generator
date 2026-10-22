//____________________________________________________________________________
/*!

\class    genie::MECGenerator

\brief    Simulate the primary MEC interaction

\author   Costas Andreopoulos <c.andreopoulos \at cern.ch>
          University of Liverpool

          Steve Dytman <dytman+ \at pitt.edu>
          Pittsburgh University

\created  Sep. 22, 2008

\cpright  Copyright (c) 2003-2025, The GENIE Collaboration
          For the full text of the license visit http://copyright.genie-mc.org
*/
//____________________________________________________________________________

#ifndef _MEC_GENERATOR_H_
#define _MEC_GENERATOR_H_

#include <TGenPhaseSpace.h>
#include <map>
#include <tuple>
#include <vector>
#include "Framework/Utils/Range1.h"

#include "Framework/EventGen/EventRecordVisitorI.h"
#include "Framework/ParticleData/PDGCodeList.h"

namespace genie {

class Interaction;
class NuclearModelI;
class XSecAlgorithmI;
class VertexGenerator;

class MECGenerator : public EventRecordVisitorI {

public :
  MECGenerator();
  MECGenerator(string config);
 ~MECGenerator();

  // implement the EventRecordVisitorI interface
  void ProcessEventRecord (GHepRecord * event) const;

  // overload the Algorithm::Configure() methods to load private data
  // members from configuration options
  void Configure(const Registry & config);
  void Configure(string config);

private:

  void    LoadConfig                        (void);
  void    AddNucleonCluster                 (GHepRecord * event) const;
  void    AddTargetRemnant                  (GHepRecord * event) const;
  void    GenerateFermiMomentum             (GHepRecord * event) const;
  void    SelectEmpiricalKinematics         (GHepRecord * event) const;
  void    AddFinalStateLepton               (GHepRecord * event) const;
  void    RecoilNucleonCluster              (GHepRecord * event) const;
  void    DecayNucleonCluster               (GHepRecord * event) const;
  void    SelectNSVLeptonKinematics         (GHepRecord * event) const;
  void    SelectSuSALeptonKinematics        (GHepRecord * event) const;
  void    GenerateNSVInitialHadrons         (GHepRecord * event) const;
  struct Valencia2020Selection {
    int channel; // pp=0, np=1, pn=2, 3p3h=3; labels are charge-conjugated for antinu
    std::vector<int> initial, final; // first final nucleon is forward for 2p2h
  };
  Valencia2020Selection SelectValencia2020LeptonKinematics(GHepRecord * event) const;
  void GenerateValencia2020Hadrons(GHepRecord * event,
                                   const Valencia2020Selection& selection) const;
  // Dense brute-force bound of the lepton (Tl, cos) density over the sampled
  // box; used when the layered scan returns zero or is exceeded.
  double Valencia2020DenseEnvelope(const Interaction & in, double TMin,
                                   double TMax, double CosthMin, double q3max) const;
  // Mark the event unphysical, count it, and skip it via a fast-forward
  // EVGThreadException. Stops the job after too many consecutive skips.
  void SkipValencia2020Event(GHepRecord * event, int flag, const std::string & reason) const;
  void PrintValencia2020Summary(void) const; // end-of-job counts (stderr)
  PDGCodeList NucleonClusterConstituents    (int pdgc)           const;

  // Helper function that computes the maximum differential cross section
  // in the kPSTlctl phase space
  double GetXSecMaxTlctl( const Interaction & inter, const Range1D_t & Tl_range, const Range1D_t & ctl_range ) const;

  mutable const XSecAlgorithmI * fXSecModel;
  mutable TGenPhaseSpace         fPhaseSpaceGenerator;
  const NuclearModelI *          fNuclModel;
  const VertexGenerator *        fValenciaVertexGenerator;

  double fSafetyFactor ; 
  int fFunctionCalls ; 
  double fRelTolerance ; // Relative tolerance 
  int fMinScanPointsTmu ; 
  int fMinScanPointsCosth ; 
  
  double fQ3Max;
  double fValenciaP[3];
  int fValenciaL[3];
  mutable std::map<std::tuple<int,int,double,const XSecAlgorithmI*>,double> fValenciaMaxima;
  // GENIE-native energy bookkeeping (Default prescription only): charge each
  // initial nucleon's nuclear-model removal energy to the hadronic system and
  // credit it to the residual nucleus, as the NSV branch does. Off by default.
  bool fValenciaRemovalEnergy;
  // Bookkeeping, printed in an end-of-job summary (destructor).
  mutable unsigned long fValenciaGenerated = 0;        // events hadronized
  mutable unsigned long fValenciaSkipped = 0;          // events discarded (all reasons)
  mutable unsigned long fValenciaSkippedHadronic = 0;  // ... of which hadronic exhaustion
  mutable unsigned long fValenciaConsecutiveSkips = 0;
  mutable unsigned long fValenciaRemovalCharged = 0;   // removal energy charged
  mutable unsigned long fValenciaRemovalFallbacks = 0; // q too small to pay it
  mutable unsigned long fValenciaEnvelopeRaises = 0;
  mutable unsigned long fValenciaDenseFallbacks = 0;
  // Multiplies new layered envelopes after any underestimate was seen
  // (continuous-energy beams never reuse a cached envelope).
  mutable double fValenciaEnvelopeInflation = 1.;
  mutable bool fValenciaRemovalIgnored = false; // option set but NuWroCompatible selected
  mutable unsigned long fValenciaNuWro3p3h = 0;      // NuWroCompatible 3p3h events
  mutable unsigned long fValenciaNuWro3p3hPauli = 0; // ... with the inherited CM Pauli flag on

  // Tolerate this maximum percent deviation above the calculated maximum cross
  // section when sampling lepton kinematics for the SuSAv2-MEC model.
  double fSuSAMaxXSecDiffTolerance;
};

}      // genie namespace
#endif // _MEC_GENERATOR_H_
