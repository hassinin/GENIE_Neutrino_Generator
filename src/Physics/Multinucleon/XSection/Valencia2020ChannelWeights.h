#ifndef GENIE_VALENCIA2020_CHANNEL_WEIGHTS_H
#define GENIE_VALENCIA2020_CHANNEL_WEIGHTS_H
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace genie { namespace utils { namespace valencia2020 {

// Preserve the signed inclusive 2p2h response. Negative partial contractions
// cannot be probabilities: distribute that rate over the positive partials
// in their original proportions. This is an exclusive sampling approximation,
// not a claim that the projected partials equal microscopic cross sections.
// 3p3h is a separate multiplicity sector and is never used to cancel 2p2h.
inline std::array<double,4> ProjectChannelWeights(std::array<double,4> raw)
{
  for (double value : raw)
    if (!std::isfinite(value))
      throw std::runtime_error("Valencia2020: nonfinite channel contraction");
  const double inclusive2 = std::max(0., raw[0]+raw[1]+raw[2]);
  double positive2 = 0.;
  for (int i=0; i<3; ++i) positive2 += std::max(0.,raw[i]);
  const double scale = positive2 > 0. ? inclusive2/positive2 : 0.;
  for (int i=0; i<3; ++i) raw[i] = std::max(0.,raw[i])*scale;
  raw[3] = std::max(0.,raw[3]);
  return raw;
}

// Installed NuWro samples same-charge first, using positive_pp/signed_2p2h,
// then splits the remainder between the positive mixed channels. A ratio
// above one means same-charge with probability one, not a larger sector rate.
inline std::array<double,4> NuWroChannelWeights(std::array<double,4> raw)
{
  for (double value : raw)
    if (!std::isfinite(value))
      throw std::runtime_error("Valencia2020: nonfinite channel contraction");
  const double two = std::max(0., raw[0]+raw[1]+raw[2]);
  const double same = std::min(two, std::max(0., raw[0]));
  const double mixed = std::max(0.,raw[1])+std::max(0.,raw[2]);
  // Form the bounded probability before multiplying: (left*np)/mixed can
  // round above left when pn<=0, producing a tiny negative pn weight.
  const double np = mixed > 0. ? std::min(two-same,
    (two-same)*(std::max(0.,raw[1])/mixed)) : 0.;
  return {{same, np, two-same-np, std::max(0.,raw[3])}};
}
}}}
#endif
