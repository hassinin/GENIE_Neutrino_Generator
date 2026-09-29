#ifndef GPU_SPLINE_TENSOR_KINEMATICS_H
#define GPU_SPLINE_TENSOR_KINEMATICS_H
#include "gpu_spline/portable_gpu.h"
#include <cfloat>
#include <cmath>

namespace gpuspline {
// Avoid subtracting two O(E^2) quantities for forward scattering.
GPU_HOST_DEVICE inline double TensorQ3Squared(double ki, double kf, double costh) {
    const double delta = ki-kf;
    return delta*delta + 2.0*ki*kf*(1.0-costh);
}

// Arithmetic at an exact support edge can differ by a few ulps between host
// and device. Interpolation already clamps to the table; admit only roundoff
// here, not a finite extension of the tensor's physical support.
GPU_HOST_DEVICE inline bool TensorInSupport(double q0, double q3, double energy,
                                            double q0min, double q0max,
                                            double q3min, double q3max) {
    const double scale = fmax(1.0, fmax(fabs(energy), fmax(fabs(q0max), fabs(q3max))));
    const double tolerance = 16.0*DBL_EPSILON*scale;
    return q0 >= q0min-tolerance && q0 <= q0max+tolerance &&
           q3 >= q3min-tolerance && q3 <= q3max+tolerance;
}
}
#endif
