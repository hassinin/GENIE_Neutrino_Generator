#include "gpu_spline/gpu_spline_engine.h"
#include <iostream>
#include <stdexcept>
#include <cstring>

extern "C" gpuError_t launch_spline_eval_kernel(
    const gpu_spline::SplineMetadata* d_metadata,
    const gpu_spline::IntervalCoeffs* d_intervals,
    const double* d_energies,
    const int* d_spline_ids,
    double* d_results,
    size_t n_queries,
    int n_splines,
    gpuStream_t stream);

namespace gpu_spline {

GpuSplineEngine::GpuSplineEngine()
    : d_metadata_(nullptr),
      d_intervals_(nullptr),
      is_device_init_(false),
      device_id_(0)
{
}

GpuSplineEngine::~GpuSplineEngine() {
    FreeDevice();
}

bool GpuSplineEngine::InitializeDevice(int device_id) {
    if (h_metadata_.empty() || h_intervals_.empty()) {
        std::cerr << "[GpuSplineEngine] Error: Cannot initialize device before loading splines!" << std::endl;
        return false;
    }

    FreeDevice();
    device_id_ = device_id;

#if !defined(CPU_ONLY)
    #if defined(__HIPCC__) || defined(__HIP__) || defined(USE_HIP)
        if (!gpuTry(hipSetDevice(device_id_))) { FreeDevice(); return false; }
    #else
        if (!gpuTry(cudaSetDevice(device_id_))) { FreeDevice(); return false; }
    #endif


    size_t meta_bytes = h_metadata_.size() * sizeof(SplineMetadata);
    size_t interval_bytes = h_intervals_.size() * sizeof(IntervalCoeffs);

    std::cout << "[GpuSplineEngine] Allocating " << (meta_bytes + interval_bytes) / (1024.0 * 1024.0)
              << " MB on GPU " << device_id_ << "..." << std::endl;

    if (!gpuTry(gpuMalloc(&d_metadata_, meta_bytes))) { FreeDevice(); return false; }
    if (!gpuTry(gpuMalloc(&d_intervals_, interval_bytes))) { FreeDevice(); return false; }

    if (!gpuTry(gpuMemcpy(d_metadata_, h_metadata_.data(), meta_bytes, gpuMemcpyHostToDevice))) { FreeDevice(); return false; }
    if (!gpuTry(gpuMemcpy(d_intervals_, h_intervals_.data(), interval_bytes, gpuMemcpyHostToDevice))) { FreeDevice(); return false; }

    is_device_init_ = true;
    std::cout << "[GpuSplineEngine] Device initialization complete." << std::endl;
#else
    std::cout << "[GpuSplineEngine] Compiled in CPU mode, using host memory tables." << std::endl;
    is_device_init_ = true;
#endif

    return true;
}

void GpuSplineEngine::FreeDevice() {
#if !defined(CPU_ONLY)
    if (d_metadata_ || d_intervals_) gpuSetDevice(device_id_);
    if (d_metadata_) {
        gpuFree(d_metadata_);
        d_metadata_ = nullptr;
    }
    if (d_intervals_) {
        gpuFree(d_intervals_);
        d_intervals_ = nullptr;
    }
#endif
    is_device_init_ = false;
}

double GpuSplineEngine::EvaluateHost(int spline_id, double energy) const {
    if (spline_id < 0 || spline_id >= static_cast<int>(h_metadata_.size())) {
        return 0.0;
    }

    const SplineMetadata& meta = h_metadata_[spline_id];
    if (energy < meta.xmin || energy > meta.xmax) {
        return 0.0;
    }

    const int offset = meta.knot_offset;
    int low = 0;
    int high = meta.nintervals - 1;

    while (low <= high) {
        int mid = (low + high) >> 1;
        double x_mid = h_intervals_[offset + mid].x;
        // ROOT TSpline3::FindX chooses the left interval at an exact knot.
        if (x_mid < energy) {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }

    int i = low - 1;
    if (i < 0) i = 0;
    if (i >= meta.nintervals) i = meta.nintervals - 1;

    const IntervalCoeffs& coeff = h_intervals_[offset + i];
    double dx = energy - coeff.x;
    double y = coeff.a + dx * (coeff.b + dx * (coeff.c + dx * coeff.d));

    return y;
}

void GpuSplineEngine::EvaluateBatchHost(const double* energies, const int* spline_ids, double* results, size_t n_queries) const {
    #pragma omp parallel for
    for (size_t i = 0; i < n_queries; ++i) {
        results[i] = EvaluateHost(spline_ids[i], energies[i]);
    }
}

void GpuSplineEngine::EvaluateBatchGpu(const double* h_energies, const int* h_spline_ids, double* h_results, size_t n_queries) const {
    if (!is_device_init_) {
        EvaluateBatchHost(h_energies, h_spline_ids, h_results, n_queries);
        return;
    }
    if (n_queries == 0) return;

#if !defined(CPU_ONLY)
    GPU_CHECK(gpuSetDevice(device_id_));
    double* d_energies = nullptr;
    int* d_spline_ids = nullptr;
    double* d_results = nullptr;

    size_t e_bytes = n_queries * sizeof(double);
    size_t id_bytes = n_queries * sizeof(int);

    GPU_CHECK(gpuMalloc(&d_energies, e_bytes));
    GPU_CHECK(gpuMalloc(&d_spline_ids, id_bytes));
    GPU_CHECK(gpuMalloc(&d_results, e_bytes));

    GPU_CHECK(gpuMemcpy(d_energies, h_energies, e_bytes, gpuMemcpyHostToDevice));
    GPU_CHECK(gpuMemcpy(d_spline_ids, h_spline_ids, id_bytes, gpuMemcpyHostToDevice));

    GPU_CHECK(launch_spline_eval_kernel(d_metadata_, d_intervals_, d_energies, d_spline_ids, d_results, n_queries, GetNumSplines(), 0));
    GPU_CHECK(gpuDeviceSynchronize());

    GPU_CHECK(gpuMemcpy(h_results, d_results, e_bytes, gpuMemcpyDeviceToHost));

    GPU_CHECK(gpuFree(d_energies));
    GPU_CHECK(gpuFree(d_spline_ids));
    GPU_CHECK(gpuFree(d_results));
#else
    EvaluateBatchHost(h_energies, h_spline_ids, h_results, n_queries);
#endif
}

void GpuSplineEngine::EvaluateDeviceData(const double* d_energies, const int* d_spline_ids, double* d_results, size_t n_queries, gpuStream_t stream) const {
    if (n_queries == 0) return;
    if (!is_device_init_) {
        throw std::runtime_error("[GpuSplineEngine] Device not initialized before EvaluateDeviceData!");
    }
#if !defined(CPU_ONLY)
    const auto device_error = gpuSetDevice(device_id_);
    if (device_error != gpuSuccess) throw std::runtime_error(gpuGetErrorString(device_error));
#endif
    const auto launch_error = launch_spline_eval_kernel(d_metadata_, d_intervals_, d_energies, d_spline_ids, d_results, n_queries, GetNumSplines(), stream);
    if (launch_error != gpuSuccess) throw std::runtime_error(gpuGetErrorString(launch_error));
}

} // namespace gpu_spline
