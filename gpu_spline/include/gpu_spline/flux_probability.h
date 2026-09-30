#ifndef GPU_SPLINE_FLUX_PROBABILITY_H
#define GPU_SPLINE_FLUX_PROBABILITY_H

#include "portable_gpu.h"

namespace gpu_spline {

// xsec is in GeV^-2; column_density is in kg/m^2. Match GENIE's
// InteractionProbability: NA * (xsec / cm2) * (column_density * 0.1) / A.
GPU_HOST_DEVICE inline double flux_interaction_probability(
    double xsec, double column_density, int A)
{
    constexpr double hbarc = 1.973269804e-16; // GeV m
    constexpr double avogadro = 6.02214179e23;
    return A > 0 ? avogadro * 1000.0 * hbarc * hbarc * xsec * column_density / A : 0.0;
}

} // namespace gpu_spline
#endif
