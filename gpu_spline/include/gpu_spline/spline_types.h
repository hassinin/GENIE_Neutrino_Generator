#ifndef GPU_SPLINE_SPLINE_TYPES_H
#define GPU_SPLINE_SPLINE_TYPES_H

#include "portable_gpu.h"
#include <string>

namespace gpu_spline {

// Spline metadata stored on host and device
struct SplineMetadata {
    int knot_offset;       // Start index in the global interval/knot arrays
    int nknots;            // Total knots (e.g. 250)
    int nintervals;         // Total intervals (nknots - 1)
    double xmin;           // Minimum valid energy
    double xmax;           // Maximum valid energy
};

// Precomputed cubic polynomial coefficients for interval [x_i, x_{i+1}]
// S(x) = a + b*(x - x_i) + c*(x - x_i)^2 + d*(x - x_i)^3
struct IntervalCoeffs {
    double x;              // Knot position x_i
    double a;              // Coefficient a (value at x_i)
    double b;              // Coefficient b (first derivative)
    double c;              // Coefficient c (half of second derivative)
    double d;              // Coefficient d (one-sixth of third derivative)
};

// Query struct for batched evaluations
struct SplineQuery {
    int spline_id;
    double energy;
};

} // namespace gpu_spline

#endif // GPU_SPLINE_SPLINE_TYPES_H
