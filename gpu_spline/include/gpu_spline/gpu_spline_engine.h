#ifndef GPU_SPLINE_GPU_SPLINE_ENGINE_H
#define GPU_SPLINE_GPU_SPLINE_ENGINE_H

#include "portable_gpu.h"
#include "spline_types.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <cstdint>

namespace gpu_spline {

class GpuSplineEngine {
public:
    GpuSplineEngine();
    ~GpuSplineEngine();

    // Disable copy
    GpuSplineEngine(const GpuSplineEngine&) = delete;
    GpuSplineEngine& operator=(const GpuSplineEngine&) = delete;

    // Load XML/XML.GZ; success invalidates GPU tables, failure preserves them.
    bool LoadSplinesFromXmlGz(const std::string& filepath);

    // Add a spline; invalidates uploaded tables. Reinitialize to resume GPU use.
    int AddSpline(const std::string& name, const std::vector<double>& x, const std::vector<double>& y);
    // A (tune, name) pair must be unique; duplicates fail without changing tables.
    int AddSpline(const std::string& name, const std::vector<double>& x,
                  const std::vector<double>& y, const std::string& tune);

    // Initialize GPU device memory and upload spline tables
    bool InitializeDevice(int device_id = 0);

    // Free GPU device allocations
    void FreeDevice();

    // Metadata queries
    int GetNumSplines() const { return static_cast<int>(spline_names_.size()); }
    int GetTotalIntervals() const { return static_cast<int>(h_intervals_.size()); }
    // Unqualified lookup: -1 means missing, -2 means ambiguous across tunes.
    int FindSplineId(const std::string& name) const;
    int FindSplineId(const std::string& name, const std::string& tune) const;
    const std::string& GetSplineName(int id) const;
    const std::string& GetSplineTune(int id) const;
    const std::vector<std::string>& GetAllSplineNames() const { return spline_names_; }
    const SplineMetadata& GetSplineMetadata(int id) const { return h_metadata_[id]; }

    // Host (CPU) single evaluation
    double EvaluateHost(int spline_id, double energy) const;

    // Host (CPU) batched evaluation (multi-threaded or single-threaded)
    void EvaluateBatchHost(const double* energies, const int* spline_ids, double* results, size_t n_queries) const;

    // GPU batch evaluation; uses current host tables when the device is
    // uninitialized, and falls back to them if the device reports an error.
    void EvaluateBatchGpu(const double* h_energies, const int* h_spline_ids, double* h_results, size_t n_queries) const;

    // Pure device-to-device evaluation (data already in GPU memory)
    void EvaluateDeviceData(const double* d_energies, const int* d_spline_ids, double* d_results, size_t n_queries, gpuStream_t stream = 0) const;

    // Direct access to device pointers if needed by external GPU workflows
    const SplineMetadata* GetDeviceMetadata() const { return d_metadata_; }
    const IntervalCoeffs* GetDeviceIntervals() const { return d_intervals_; }

    bool IsDeviceInitialized() const { return is_device_init_; }
    // Changes on successful table replacement. Appending preserves existing
    // IDs/values; failed loads and device reuploads preserve this generation.
    std::uint64_t TableGeneration() const { return table_generation_; }

private:
    std::vector<std::string> spline_names_;
    std::vector<std::string> spline_tunes_;
    std::unordered_map<std::string, int> name_to_id_;
    std::unordered_map<std::string, std::unordered_map<std::string, int>> tune_to_ids_;

    // Host tables
    std::vector<SplineMetadata> h_metadata_;
    std::vector<IntervalCoeffs> h_intervals_;

    // Device pointers
    SplineMetadata* d_metadata_;
    IntervalCoeffs* d_intervals_;
    bool is_device_init_;
    int device_id_;
    std::uint64_t table_generation_ = 0;
};

} // namespace gpu_spline

#endif // GPU_SPLINE_GPU_SPLINE_ENGINE_H
