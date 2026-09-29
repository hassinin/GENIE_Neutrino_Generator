#ifndef GPU_SPLINE_EVAL_DEVICE_H
#define GPU_SPLINE_EVAL_DEVICE_H

#include "portable_gpu.h"
#include "spline_types.h"

namespace gpu_spline {

GPU_DEVICE inline double evaluate_single_spline(
    const SplineMetadata* __restrict__ metadata_table,
    const IntervalCoeffs* __restrict__ interval_table,
    int n_splines,
    int spline_id,
    double energy)
{
    if (spline_id < 0 || spline_id >= n_splines) return 0.0;
    const SplineMetadata meta = metadata_table[spline_id];

    // Out-of-bounds check matching GENIE.
    if (energy < meta.xmin || energy > meta.xmax) {
        return 0.0;
    }

    const int offset = meta.knot_offset;
    int low = 0;
    int high = meta.nintervals - 1;

    // Fast binary search across knot intervals
    while (low <= high) {
        int mid = (low + high) >> 1;
        double x_mid = interval_table[offset + mid].x;
        // Match ROOT's left interval at exact interior knots.
        if (x_mid < energy) {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }

    int i = low - 1;
    if (i < 0) i = 0;
    if (i >= meta.nintervals) i = meta.nintervals - 1;

    const IntervalCoeffs coeff = interval_table[offset + i];
    double dx = energy - coeff.x;

    // Horner's method / Fused Multiply-Add (FMA) for cubic polynomial:
    // y = a + dx * (b + dx * (c + dx * d))
    double y = coeff.a + dx * (coeff.b + dx * (coeff.c + dx * coeff.d));

    // GENIE Spline::Evaluate also returns negative interpolated values.
    return y;
}

} // namespace gpu_spline

#endif // GPU_SPLINE_EVAL_DEVICE_H
