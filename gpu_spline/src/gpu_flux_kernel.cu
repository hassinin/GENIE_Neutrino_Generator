#include "gpu_spline/portable_gpu.h"
#include "gpu_spline/spline_types.h"
#include "gpu_spline/gpu_flux_preselector.h"
#include "gpu_spline/spline_eval_device.h"
#include "gpu_spline/flux_probability.h"

namespace gpu_spline {

// Helper function to map PDG code to spline ID
GPU_DEVICE inline int get_spline_id_for_flavor(const MaterialConfig& mat, int nupdg) {
    switch (nupdg) {
        case  12: return mat.spline_id_nue;
        case -12: return mat.spline_id_nuebar;
        case  14: return mat.spline_id_numu;
        case -14: return mat.spline_id_numubar;
        case  16: return mat.spline_id_nutau;
        case -16: return mat.spline_id_nutaubar;
        default:  return -1;
    }
}

GPU_GLOBAL void flux_preselect_kernel(
    const SplineMetadata* __restrict__ d_metadata,
    const IntervalCoeffs* __restrict__ d_intervals,
    const MaterialConfig* __restrict__ d_materials,
    int n_materials,
    int n_splines,
    double glob_pmax,
    const double* __restrict__ d_energies,
    const int* __restrict__ d_nupdgs,
    const double* __restrict__ d_rndms,
    double* __restrict__ d_psums,
    int* __restrict__ d_accepted_indices,
    int* __restrict__ d_accepted_count,
    size_t n_rays)
{
    size_t idx = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    if (idx >= n_rays) return;

    double E = d_energies[idx];
    int nupdg = d_nupdgs[idx];
    double R = d_rndms[idx];

    double psum = 0.0;

    for (int m = 0; m < n_materials; ++m) {
        const MaterialConfig mat = d_materials[m];
        if (mat.max_pl <= 0.0) continue;

        int sp_id = get_spline_id_for_flavor(mat, nupdg);
        if (sp_id < 0) continue;

        double xsec = evaluate_single_spline(d_metadata, d_intervals, n_splines, sp_id, E);
        if (xsec > 0.0) {
            double prob = flux_interaction_probability(xsec, mat.max_pl, mat.A);
            double probn = prob / glob_pmax;
            psum += probn;
        }
    }

    d_psums[idx] = psum;

    // Interaction test: if R < Psum, ray interacts!
    if (R < psum) {
#if defined(__HIPCC__) || defined(__HIP__) || defined(__CUDACC__) || defined(__NVCC__)
        int out_idx = atomicAdd(d_accepted_count, 1);
        if (out_idx < n_rays) {
            d_accepted_indices[out_idx] = static_cast<int>(idx);
        }
#endif
    }
}

extern "C" void launch_flux_preselect_kernel(
    const SplineMetadata* d_metadata,
    const IntervalCoeffs* d_intervals,
    const MaterialConfig* d_materials,
    int n_materials,
    int n_splines,
    double glob_pmax,
    const double* d_energies,
    const int* d_nupdgs,
    const double* d_rndms,
    double* d_psums,
    int* d_accepted_indices,
    int* d_accepted_count,
    size_t n_rays,
    gpuStream_t stream)
{
    if (n_rays == 0) return;

    int block_size = 256;
    int num_blocks = static_cast<int>((n_rays + block_size - 1) / block_size);

#if defined(__HIPCC__) || defined(__HIP__) || defined(__CUDACC__) || defined(__NVCC__)
    if (stream) {
        flux_preselect_kernel<<<num_blocks, block_size, 0, stream>>>(
            d_metadata, d_intervals, d_materials, n_materials, n_splines, glob_pmax,
            d_energies, d_nupdgs, d_rndms, d_psums, d_accepted_indices, d_accepted_count, n_rays);
    } else {
        flux_preselect_kernel<<<num_blocks, block_size>>>(
            d_metadata, d_intervals, d_materials, n_materials, n_splines, glob_pmax,
            d_energies, d_nupdgs, d_rndms, d_psums, d_accepted_indices, d_accepted_count, n_rays);
    }
    GPU_CHECK(gpuGetLastError());
#else
    // CPU fallback
    int count = 0;
    for (size_t idx = 0; idx < n_rays; ++idx) {
        double E = d_energies[idx];
        int nupdg = d_nupdgs[idx];
        double R = d_rndms[idx];

        double psum = 0.0;
        for (int m = 0; m < n_materials; ++m) {
            const MaterialConfig& mat = d_materials[m];
            int sp_id = get_spline_id_for_flavor(mat, nupdg);
            if (sp_id < 0) continue;

            double xsec = evaluate_single_spline(d_metadata, d_intervals, n_splines, sp_id, E);
            if (xsec > 0.0) {
                double prob = flux_interaction_probability(xsec, mat.max_pl, mat.A);
                psum += prob / glob_pmax;
            }
        }
        d_psums[idx] = psum;
        if (R < psum) {
            d_accepted_indices[count++] = static_cast<int>(idx);
        }
    }
    *d_accepted_count = count;
#endif
}

} // namespace gpu_spline
