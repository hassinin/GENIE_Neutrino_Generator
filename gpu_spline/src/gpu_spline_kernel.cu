#include "gpu_spline/portable_gpu.h"
#include "gpu_spline/spline_types.h"
#include "gpu_spline/spline_eval_device.h"

namespace gpu_spline {

GPU_GLOBAL void spline_eval_kernel(
    const SplineMetadata* __restrict__ d_metadata,
    const IntervalCoeffs* __restrict__ d_intervals,
    const double* __restrict__ d_energies,
    const int* __restrict__ d_spline_ids,
    double* __restrict__ d_results,
    size_t n_queries, int n_splines)
{
    size_t idx = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    if (idx < n_queries) {
        int sp_id = d_spline_ids[idx];
        double e = d_energies[idx];
        d_results[idx] = evaluate_single_spline(d_metadata, d_intervals, n_splines, sp_id, e);
    }
}

// Host launcher function callable from C++
extern "C" gpuError_t launch_spline_eval_kernel(
    const SplineMetadata* d_metadata,
    const IntervalCoeffs* d_intervals,
    const double* d_energies,
    const int* d_spline_ids,
    double* d_results,
    size_t n_queries,
    int n_splines,
    gpuStream_t stream)
{
    if (n_queries == 0) return gpuSuccess;

    int block_size = 256;
    int num_blocks = static_cast<int>((n_queries + block_size - 1) / block_size);

#if defined(__HIPCC__) || defined(__HIP__) || defined(__CUDACC__) || defined(__NVCC__)
    if (stream) {
        spline_eval_kernel<<<num_blocks, block_size, 0, stream>>>(
            d_metadata, d_intervals, d_energies, d_spline_ids, d_results, n_queries, n_splines);
    } else {
        spline_eval_kernel<<<num_blocks, block_size>>>(
            d_metadata, d_intervals, d_energies, d_spline_ids, d_results, n_queries, n_splines);
    }
    return gpuGetLastError();
#else
    // CPU fallback
    #pragma omp parallel for
    for (size_t idx = 0; idx < n_queries; ++idx) {
        int sp_id = d_spline_ids[idx];
        double e = d_energies[idx];
        d_results[idx] = evaluate_single_spline(d_metadata, d_intervals, n_splines, sp_id, e);
    }
    return gpuSuccess;
#endif
}

} // namespace gpu_spline
