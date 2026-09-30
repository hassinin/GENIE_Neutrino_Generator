#include "gpu_spline/gpu_spline_c_api.h"
#include "gpu_spline/gpu_spline_engine.h"
#include <cstdio>
#include <exception>

namespace {
thread_local char device_error[1024] = {};
}

gpu_spline_handle_t gpu_spline_create() {
    return new (std::nothrow) gpu_spline::GpuSplineEngine();
}

void gpu_spline_destroy(gpu_spline_handle_t handle) {
    if (handle) {
        delete static_cast<gpu_spline::GpuSplineEngine*>(handle);
    }
}

int gpu_spline_load_xml(gpu_spline_handle_t handle, const char* filepath) {
    if (!handle || !filepath) return 0;
    auto* engine = static_cast<gpu_spline::GpuSplineEngine*>(handle);
    return engine->LoadSplinesFromXmlGz(filepath) ? 1 : 0;
}

int gpu_spline_init_gpu(gpu_spline_handle_t handle, int device_id) {
    if (!handle) return 0;
    auto* engine = static_cast<gpu_spline::GpuSplineEngine*>(handle);
    return engine->InitializeDevice(device_id) ? 1 : 0;
}

void gpu_spline_free_gpu(gpu_spline_handle_t handle) {
    if (!handle) return;
    auto* engine = static_cast<gpu_spline::GpuSplineEngine*>(handle);
    engine->FreeDevice();
}

int gpu_spline_get_num_splines(gpu_spline_handle_t handle) {
    if (!handle) return 0;
    auto* engine = static_cast<gpu_spline::GpuSplineEngine*>(handle);
    return engine->GetNumSplines();
}

int gpu_spline_find_id(gpu_spline_handle_t handle, const char* name) {
    if (!handle || !name) return -1;
    auto* engine = static_cast<gpu_spline::GpuSplineEngine*>(handle);
    return engine->FindSplineId(name);
}

int gpu_spline_find_id_for_tune(gpu_spline_handle_t handle, const char* name, const char* tune) {
    if (!handle || !name || !tune) return -1;
    return static_cast<gpu_spline::GpuSplineEngine*>(handle)->FindSplineId(name, tune);
}

const char* gpu_spline_get_tune(gpu_spline_handle_t handle, int spline_id) {
    if (!handle) return "";
    return static_cast<gpu_spline::GpuSplineEngine*>(handle)->GetSplineTune(spline_id).c_str();
}

const char* gpu_spline_get_name(gpu_spline_handle_t handle, int spline_id) {
    if (!handle) return "";
    auto* engine = static_cast<gpu_spline::GpuSplineEngine*>(handle);
    return engine->GetSplineName(spline_id).c_str();
}

double gpu_spline_eval_single_cpu(gpu_spline_handle_t handle, int spline_id, double energy) {
    if (!handle) return 0.0;
    auto* engine = static_cast<gpu_spline::GpuSplineEngine*>(handle);
    return engine->EvaluateHost(spline_id, energy);
}

void gpu_spline_eval_batch_cpu(gpu_spline_handle_t handle, const double* energies, const int* spline_ids, double* results, size_t n) {
    if (!handle || !energies || !spline_ids || !results || n == 0) return;
    auto* engine = static_cast<gpu_spline::GpuSplineEngine*>(handle);
    engine->EvaluateBatchHost(energies, spline_ids, results, n);
}

void gpu_spline_eval_batch_gpu(gpu_spline_handle_t handle, const double* energies, const int* spline_ids, double* results, size_t n) {
    if (!handle || !energies || !spline_ids || !results || n == 0) return;
    auto* engine = static_cast<gpu_spline::GpuSplineEngine*>(handle);
    // No exception may cross into C callers (e.g. Python ctypes). Device
    // errors already fall back inside EvaluateBatchGpu; this guards the rest.
    try {
        engine->EvaluateBatchGpu(energies, spline_ids, results, n);
    } catch (...) {
        engine->EvaluateBatchHost(energies, spline_ids, results, n);
    }
}

void gpu_spline_eval_device(gpu_spline_handle_t handle, const double* d_energies, const int* d_spline_ids, double* d_results, size_t n, void* stream) {
    gpu_spline_eval_device_checked(handle, d_energies, d_spline_ids, d_results, n, stream);
}

int gpu_spline_eval_device_checked(gpu_spline_handle_t handle, const double* d_energies, const int* d_spline_ids, double* d_results, size_t n, void* stream) {
    device_error[0] = '\0';
    if (n == 0) return 1;
    if (!handle || !d_energies || !d_spline_ids || !d_results) {
        std::snprintf(device_error, sizeof(device_error), "Null handle or device buffer");
        return 0;
    }
    try {
        auto* engine = static_cast<gpu_spline::GpuSplineEngine*>(handle);
        engine->EvaluateDeviceData(d_energies, d_spline_ids, d_results, n, static_cast<gpuStream_t>(stream));
        return 1;
    } catch (const std::exception& error) {
        std::snprintf(device_error, sizeof(device_error), "%s", error.what());
    } catch (...) {
        std::snprintf(device_error, sizeof(device_error), "Unknown device evaluation error");
    }
    return 0;
}

const char* gpu_spline_device_last_error(void) {
    return device_error;
}
