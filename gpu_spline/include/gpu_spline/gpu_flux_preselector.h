#ifndef GPU_SPLINE_GPU_FLUX_PRESELECTOR_H
#define GPU_SPLINE_GPU_FLUX_PRESELECTOR_H

#include "portable_gpu.h"
#include "spline_types.h"
#include "gpu_spline_engine.h"
#include <vector>
#include <queue>
#include <memory>
#include <random>

namespace gpu_spline {

struct MaterialConfig {
    int target_pdg;       // Target PDG (e.g. 1000180400 for Ar40)
    int A;                // Mass number A
    double max_pl;        // Density-weighted max path length (kg/m^2)
    // Mapping from neutrino PDG to spline ID in GpuSplineEngine
    // Supported flavors: 12 (nue), -12 (nuebar), 14 (numu), -14 (numubar), 16 (nutau), -16 (nutaubar)
    int spline_id_nue;
    int spline_id_nuebar;
    int spline_id_numu;
    int spline_id_numubar;
    int spline_id_nutau;
    int spline_id_nutaubar;
};

// Auxiliary ray kinematics stored on host for accepted candidates
struct RayAuxData {
    double px, py, pz, E;
    double x, y, z, t;
    double weight;
    long int index;
};

// A flux candidate ray that was accepted by GPU preselection
struct AcceptedRay {
    int nupdg;
    double energy;
    double px, py, pz, E;
    double x, y, z, t;
    double weight;
    long int index;
    double R;       // Random number thrown
    double Psum;    // Precomputed max interaction probability
    long int thrown_before_this; // Number of rejected rays thrown before this accepted ray
};

class GpuFluxPreselector {
public:
    GpuFluxPreselector(GpuSplineEngine* engine);
    ~GpuFluxPreselector();

    GpuFluxPreselector(const GpuFluxPreselector&) = delete;
    GpuFluxPreselector& operator=(const GpuFluxPreselector&) = delete;

    // Physics configuration changes discard queued rays and reset statistics.
    // Adding materials also requires InitializeDevice before filtering again.
    void AddMaterial(const MaterialConfig& mat);
    // After engine table replacement, clear and rebuild mappings using current
    // spline IDs, then InitializeDevice. Stale mappings cannot be reuploaded.
    void ClearMaterials();
    void SetGlobalPmax(double pmax);
    void SetBatchSize(size_t batch_size);
    size_t GetBatchSize() const { return batch_size_; }

    // Start a new stream: discard queued rays, trailing rejects and statistics.
    bool InitializeDevice(int device_id = 0);
    void FreeDevice();

    // Check if there are accepted candidates ready
    // Stream access throws after table replacement until mappings are cleared.
    bool HasCandidates() const { RequireCurrentSplines(); return !candidate_queue_.empty(); }
    size_t QueueSize() const { RequireCurrentSplines(); return candidate_queue_.size(); }

    // Retrieve next accepted candidate
    AcceptedRay PopCandidate();

    // Run GPU preselection on a batch of rays
    // Energies, NuPDGs, Rndms: arrays of size n_rays
    // Returns number of accepted rays
    size_t FilterRayBatch(
        const double* h_energies,
        const int* h_nupdgs,
        const double* h_rndms,
        const RayAuxData* h_aux_data,
        size_t n_rays);

    // Flushes any remaining rejected rays at the tail of the stream
    long int FlushTrailingRejected() {
        RequireCurrentSplines();
        long int r = trailing_rejected_;
        trailing_rejected_ = 0;
        return r;
    }
    long int GetTrailingRejected() const { RequireCurrentSplines(); return trailing_rejected_; }

    // Total statistics
    long int GetTotalRaysThrown() const { RequireCurrentSplines(); return total_rays_thrown_; }
    long int GetTotalRaysAccepted() const { RequireCurrentSplines(); return total_rays_accepted_; }

private:
    bool HasCurrentSplines() const;
    void RequireCurrentSplines() const;
    void ResetStream();
    bool InitializeDeviceStorage(int device_id);
    void FreeDeviceStorage();

    GpuSplineEngine* engine_;
    std::uint64_t spline_generation_;
    std::vector<MaterialConfig> materials_;
    double glob_pmax_;
    size_t batch_size_;

    std::queue<AcceptedRay> candidate_queue_;
    long int total_rays_thrown_;
    long int total_rays_accepted_;
    long int trailing_rejected_;

    // Device allocations for material tables
    MaterialConfig* d_materials_;
    size_t n_materials_;
    bool is_device_init_;
    int device_id_;

    // Device batch buffers
    double* d_energies_;
    int* d_nupdgs_;
    double* d_rndms_;
    double* d_psums_;
    int* d_accepted_indices_;
    int* d_accepted_count_;
    size_t allocated_batch_capacity_;
};

} // namespace gpu_spline

#endif // GPU_SPLINE_GPU_FLUX_PRESELECTOR_H
