#include "gpu_spline/gpu_flux_preselector.h"
#include "gpu_spline/flux_probability.h"
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <algorithm>

namespace gpu_spline {

GpuFluxPreselector::GpuFluxPreselector(GpuSplineEngine* engine)
    : engine_(engine),
      spline_generation_(engine ? engine->TableGeneration() : 0),
      glob_pmax_(1.0),
      batch_size_(50000),
      total_rays_thrown_(0),
      total_rays_accepted_(0),
      trailing_rejected_(0),
      d_materials_(nullptr),
      n_materials_(0),
      is_device_init_(false),
      device_id_(0),
      d_energies_(nullptr),
      d_nupdgs_(nullptr),
      d_rndms_(nullptr),
      d_psums_(nullptr),
      d_accepted_indices_(nullptr),
      d_accepted_count_(nullptr),
      allocated_batch_capacity_(0)
{
}

GpuFluxPreselector::~GpuFluxPreselector() {
    FreeDevice();
}

void GpuFluxPreselector::AddMaterial(const MaterialConfig& mat) {
    RequireCurrentSplines();
    FreeDevice();
    if (engine_) spline_generation_ = engine_->TableGeneration();
    materials_.push_back(mat);
}

void GpuFluxPreselector::ClearMaterials() {
    FreeDevice();
    materials_.clear();
    if (engine_) spline_generation_ = engine_->TableGeneration();
}

bool GpuFluxPreselector::HasCurrentSplines() const {
    return materials_.empty() || (engine_ && spline_generation_ == engine_->TableGeneration());
}

void GpuFluxPreselector::RequireCurrentSplines() const {
    if (!HasCurrentSplines()) {
        throw std::runtime_error("[GpuFluxPreselector] Spline tables replaced: call ClearMaterials, rebuild material mappings with current spline IDs, and InitializeDevice before using the stream");
    }
}

void GpuFluxPreselector::ResetStream() {
    std::queue<AcceptedRay>().swap(candidate_queue_);
    total_rays_thrown_ = 0;
    total_rays_accepted_ = 0;
    trailing_rejected_ = 0;
}

void GpuFluxPreselector::SetGlobalPmax(double pmax) {
    if (!std::isfinite(pmax) || pmax <= 0.) {
        throw std::invalid_argument("Flux probability scale must be finite and positive");
    }
    if (pmax != glob_pmax_) ResetStream();
    glob_pmax_ = pmax;
}

void GpuFluxPreselector::SetBatchSize(size_t batch_size) {
    if (batch_size == 0) throw std::invalid_argument("Flux batch size must be positive");
    batch_size_ = batch_size;
}

bool GpuFluxPreselector::InitializeDevice(int device_id) {
    ResetStream();
    return InitializeDeviceStorage(device_id);
}

bool GpuFluxPreselector::InitializeDeviceStorage(int device_id) {
    FreeDeviceStorage();
    if (!HasCurrentSplines()) {
        std::cerr << "[GpuFluxPreselector] Spline tables replaced: ClearMaterials and rebuild mappings before initialization" << std::endl;
        return false;
    }
    if (!engine_ || !engine_->IsDeviceInitialized()) {
        std::cerr << "[GpuFluxPreselector] Error: GpuSplineEngine must be initialized before flux preselector!" << std::endl;
        return false;
    }

    device_id_ = device_id;

    n_materials_ = materials_.size();
    if (n_materials_ == 0) {
        std::cerr << "[GpuFluxPreselector] Warning: No materials registered!" << std::endl;
        return false;
    }

#if !defined(CPU_ONLY)
    if (!gpuTry(gpuSetDevice(device_id_))) { FreeDeviceStorage(); return false; }

    // Allocate material table on GPU
    size_t mat_bytes = n_materials_ * sizeof(MaterialConfig);
    if (!gpuTry(gpuMalloc(&d_materials_, mat_bytes))) { FreeDeviceStorage(); return false; }
    if (!gpuTry(gpuMemcpy(d_materials_, materials_.data(), mat_bytes, gpuMemcpyHostToDevice))) { FreeDeviceStorage(); return false; }

    // Allocate batch buffers
    allocated_batch_capacity_ = batch_size_;
    if (!gpuTry(gpuMalloc(&d_energies_, allocated_batch_capacity_ * sizeof(double)))) { FreeDeviceStorage(); return false; }
    if (!gpuTry(gpuMalloc(&d_nupdgs_, allocated_batch_capacity_ * sizeof(int)))) { FreeDeviceStorage(); return false; }
    if (!gpuTry(gpuMalloc(&d_rndms_, allocated_batch_capacity_ * sizeof(double)))) { FreeDeviceStorage(); return false; }
    if (!gpuTry(gpuMalloc(&d_psums_, allocated_batch_capacity_ * sizeof(double)))) { FreeDeviceStorage(); return false; }
    if (!gpuTry(gpuMalloc(&d_accepted_indices_, allocated_batch_capacity_ * sizeof(int)))) { FreeDeviceStorage(); return false; }
    if (!gpuTry(gpuMalloc(&d_accepted_count_, sizeof(int)))) { FreeDeviceStorage(); return false; }

    is_device_init_ = true;
    std::cout << "[GpuFluxPreselector] Initialized GPU preselector with " << n_materials_
              << " materials and batch capacity " << allocated_batch_capacity_ << " rays." << std::endl;
#else
    // This class requires a real GPU; let GENIE use its CPU flux driver.
    return false;
#endif

    return true;
}

void GpuFluxPreselector::FreeDevice() {
    ResetStream();
    FreeDeviceStorage();
}

void GpuFluxPreselector::FreeDeviceStorage() {
#if !defined(CPU_ONLY)
    if (d_materials_ || d_energies_) (void)gpuSetDevice(device_id_);
    if (d_materials_) { (void)gpuFree(d_materials_); d_materials_ = nullptr; }
    if (d_energies_) { (void)gpuFree(d_energies_); d_energies_ = nullptr; }
    if (d_nupdgs_) { (void)gpuFree(d_nupdgs_); d_nupdgs_ = nullptr; }
    if (d_rndms_) { (void)gpuFree(d_rndms_); d_rndms_ = nullptr; }
    if (d_psums_) { (void)gpuFree(d_psums_); d_psums_ = nullptr; }
    if (d_accepted_indices_) { (void)gpuFree(d_accepted_indices_); d_accepted_indices_ = nullptr; }
    if (d_accepted_count_) { (void)gpuFree(d_accepted_count_); d_accepted_count_ = nullptr; }
#endif
    is_device_init_ = false;
}

AcceptedRay GpuFluxPreselector::PopCandidate() {
    RequireCurrentSplines();
    if (candidate_queue_.empty()) {
        throw std::runtime_error("[GpuFluxPreselector] Attempted to pop candidate from empty queue!");
    }
    AcceptedRay ray = candidate_queue_.front();
    candidate_queue_.pop();
    return ray;
}

size_t GpuFluxPreselector::FilterRayBatch(
    const double* h_energies,
    const int* h_nupdgs,
    const double* h_rndms,
    const RayAuxData* h_aux_data,
    size_t n_rays)
{
    RequireCurrentSplines();
    if (!is_device_init_ || !engine_ || !engine_->IsDeviceInitialized()) {
        throw std::runtime_error("[GpuFluxPreselector] Device not initialized before FilterRayBatch!");
    }
    if (n_rays == 0) return 0;

    // Resize device buffer if batch exceeds current capacity
    if (n_rays > allocated_batch_capacity_) {
        // Capacity growth does not change physics or start a new ray stream.
        batch_size_ = n_rays;
        if (!InitializeDeviceStorage(device_id_)) {
            throw GpuDeviceError(GPU_SPLINE_UNKNOWN_ERROR, "[GpuFluxPreselector] Failed to resize device buffers");
        }
    }

#if !defined(CPU_ONLY)
    GPU_CHECK(gpuSetDevice(device_id_));
    // Upload batch to GPU
    GPU_CHECK(gpuMemcpy(d_energies_, h_energies, n_rays * sizeof(double), gpuMemcpyHostToDevice));
    GPU_CHECK(gpuMemcpy(d_nupdgs_, h_nupdgs, n_rays * sizeof(int), gpuMemcpyHostToDevice));
    GPU_CHECK(gpuMemcpy(d_rndms_, h_rndms, n_rays * sizeof(double), gpuMemcpyHostToDevice));

    int zero = 0;
    GPU_CHECK(gpuMemcpy(d_accepted_count_, &zero, sizeof(int), gpuMemcpyHostToDevice));

    // Launch GPU filter kernel
    GPU_CHECK(launch_flux_preselect_kernel(
        engine_->GetDeviceMetadata(),
        engine_->GetDeviceIntervals(),
        d_materials_,
        static_cast<int>(n_materials_),
        engine_->GetNumSplines(),
        glob_pmax_,
        d_energies_,
        d_nupdgs_,
        d_rndms_,
        d_psums_,
        d_accepted_indices_,
        d_accepted_count_,
        n_rays,
        0));

    GPU_CHECK(gpuDeviceSynchronize());

    int h_accepted_count = 0;
    GPU_CHECK(gpuMemcpy(&h_accepted_count, d_accepted_count_, sizeof(int), gpuMemcpyDeviceToHost));

    std::vector<int> h_indices(h_accepted_count);
    std::vector<double> h_psums;
    if (h_accepted_count > 0) {
        GPU_CHECK(gpuMemcpy(h_indices.data(), d_accepted_indices_, h_accepted_count * sizeof(int), gpuMemcpyDeviceToHost));
        // Sort indices to ensure strictly ascending order of beam rays
        std::sort(h_indices.begin(), h_indices.end());
        h_psums.resize(n_rays);
        GPU_CHECK(gpuMemcpy(h_psums.data(), d_psums_, n_rays * sizeof(double), gpuMemcpyDeviceToHost));
    }
    // All device work is done; only now is the stream state changed.
    return EnqueueAccepted(h_indices, h_psums.data(), h_energies, h_nupdgs, h_rndms, h_aux_data, n_rays);
#else
    return 0;
#endif
}

size_t GpuFluxPreselector::FilterRayBatchHost(
    const double* h_energies,
    const int* h_nupdgs,
    const double* h_rndms,
    const RayAuxData* h_aux_data,
    size_t n_rays)
{
    RequireCurrentSplines();
    if (n_rays == 0) return 0;
    if (!engine_) throw std::runtime_error("[GpuFluxPreselector] No spline engine for host preselection");

    // Mirrors flux_preselect_kernel: same materials in the same order, same
    // skip rules and the same probability formula.
    std::vector<double> psums(n_rays, 0.0);
    std::vector<int> accepted;
    for (size_t idx = 0; idx < n_rays; ++idx) {
        double psum = 0.0;
        for (const MaterialConfig& mat : materials_) {
            if (mat.max_pl <= 0.0) continue;
            const int sp_id = get_spline_id_for_flavor(mat, h_nupdgs[idx]);
            if (sp_id < 0) continue;
            const double xsec = engine_->EvaluateHost(sp_id, h_energies[idx]);
            if (xsec > 0.0) psum += flux_interaction_probability(xsec, mat.max_pl, mat.A) / glob_pmax_;
        }
        psums[idx] = psum;
        if (h_rndms[idx] < psum) accepted.push_back(static_cast<int>(idx));
    }
    return EnqueueAccepted(accepted, psums.data(), h_energies, h_nupdgs, h_rndms, h_aux_data, n_rays);
}

size_t GpuFluxPreselector::EnqueueAccepted(
    const std::vector<int>& accepted,
    const double* psums,
    const double* h_energies,
    const int* h_nupdgs,
    const double* h_rndms,
    const RayAuxData* h_aux_data,
    size_t n_rays)
{
    if (!accepted.empty()) {
        long int prev_idx = -1 - trailing_rejected_;
        for (int ray_idx : accepted) {
            AcceptedRay ray;
            ray.nupdg = h_nupdgs[ray_idx];
            ray.energy = h_energies[ray_idx];
            if (h_aux_data) {
                ray.px = h_aux_data[ray_idx].px;
                ray.py = h_aux_data[ray_idx].py;
                ray.pz = h_aux_data[ray_idx].pz;
                ray.E  = h_aux_data[ray_idx].E;
                ray.x  = h_aux_data[ray_idx].x;
                ray.y  = h_aux_data[ray_idx].y;
                ray.z  = h_aux_data[ray_idx].z;
                ray.t  = h_aux_data[ray_idx].t;
                ray.weight = h_aux_data[ray_idx].weight;
                ray.index  = h_aux_data[ray_idx].index;
            } else {
                ray.px = 0.0; ray.py = 0.0; ray.pz = h_energies[ray_idx]; ray.E = h_energies[ray_idx];
                ray.x = 0.0; ray.y = 0.0; ray.z = 0.0; ray.t = 0.0;
                ray.weight = 1.0;
                ray.index = ray_idx;
            }
            ray.R = h_rndms[ray_idx];
            ray.Psum = psums[ray_idx];
            ray.thrown_before_this = (ray_idx - prev_idx - 1);
            prev_idx = ray_idx;

            candidate_queue_.push(ray);
        }

        trailing_rejected_ = (n_rays - 1 - accepted.back());
    } else {
        trailing_rejected_ += n_rays;
    }

    total_rays_thrown_ += n_rays;
    total_rays_accepted_ += accepted.size();

    return accepted.size();
}

} // namespace gpu_spline
