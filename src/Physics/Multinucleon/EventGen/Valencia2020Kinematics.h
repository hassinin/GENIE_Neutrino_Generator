// Small, independently testable pieces of arXiv:2411.11523, Appendices A/B.
#ifndef GENIE_VALENCIA2020_KINEMATICS_H
#define GENIE_VALENCIA2020_KINEMATICS_H
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <TRandom.h>

namespace genie { namespace utils { namespace valencia2020 {

// Unnormalised density on |cos(theta*)|/kappa in [0,1]. Its maximum is 1.
inline double AngularShape(double y, double P, int l)
{
  return (P >= 0. ? 1.-P : 1.) + P*std::pow(y, l);
}

inline double SampleForwardCosine(TRandom& rng, double P, int l,
                                  double lower, double kappa)
{
  if (l < 1 || std::abs(P) > 1. || lower < 0. || kappa > 1. || lower >= kappa)
    throw std::invalid_argument("Invalid Valencia2020 angular support/parameters");
  for (int i = 0; i < 100000; ++i) {
    const double c = lower + (kappa-lower)*rng.Rndm();
    if (rng.Rndm() < AngularShape(c/kappa, P, l)) return c;
  }
  throw std::runtime_error("Valencia2020 angular sampling exhausted");
}

// First daughter is forward. Intersect BOTH species' exact lab Fermi cuts.
inline bool PauliAngularRange(double beta, double pstar, double e1, double e2,
                              double ef1, double ef2, double& lower, double& upper)
{
  lower = 0.; upper = 1.;
  const double gamma = 1./std::sqrt(1.-beta*beta);
  const double denom = gamma*beta*pstar;
  if (denom < 1.e-14) return gamma*e1 > ef1 && gamma*e2 > ef2;
  lower = std::max(0., (ef1-gamma*e1)/denom);
  upper = std::min(1., (gamma*e2-ef2)/denom);
  return lower < upper;
}

inline double Choose(int n, int k)
{
  if (k < 0 || k > n) return 0.;
  double value = 1.;
  for (int i = 1; i <= k; ++i) value *= double(n-i+1)/i;
  return value;
}

// Eq. (11): original target Z,N, not the charge-exchanged nucleus.
inline double ThreeBodyChargeWeight(int Z, int N, int finalZ, bool antinu)
{
  const int initialZ = finalZ + (antinu ? 1 : -1);
  if (initialZ < 0 || initialZ > 3 || initialZ > Z || 3-initialZ > N) return 0.;
  return Choose(Z, finalZ)*Choose(N, 3-finalZ);
}

}}} // genie::utils::valencia2020
#endif
