#ifndef GPU_HADRON_TENSOR_H
#define GPU_HADRON_TENSOR_H

#include <string>
#include <vector>
#include "gpu_spline/portable_gpu.h"

namespace gpuspline {

/// 5 independent elements of the symmetric + antisymmetric lab-frame hadron tensor
struct GpuHadronTensorEntry {
    double W00;
    double ReW0z;
    double Wxx;
    double ImWxy;
    double Wzz;

    GPU_HOST_DEVICE GpuHadronTensorEntry()
        : W00(0.0), ReW0z(0.0), Wxx(0.0), ImWxy(0.0), Wzz(0.0) {}

    GPU_HOST_DEVICE GpuHadronTensorEntry(double w00, double rew0z, double wxx, double imwxy, double wzz)
        : W00(w00), ReW0z(rew0z), Wxx(wxx), ImWxy(imwxy), Wzz(wzz) {}

    GPU_HOST_DEVICE inline GpuHadronTensorEntry operator*(double s) const {
        return GpuHadronTensorEntry(W00*s, ReW0z*s, Wxx*s, ImWxy*s, Wzz*s);
    }

    GPU_HOST_DEVICE inline GpuHadronTensorEntry operator+(const GpuHadronTensorEntry& o) const {
        return GpuHadronTensorEntry(W00+o.W00, ReW0z+o.ReW0z, Wxx+o.Wxx, ImWxy+o.ImWxy, Wzz+o.Wzz);
    }
};

/// Device-ready tensor metadata and memory pointers
struct GpuHadronTensorData {
    int Z;
    int A;
    int target_pdg;
    int num_q0;
    int num_q3;
    int q0_flag;     // 0 = regular step, 1 = explicit points
    int q3_flag;     // 0 = regular step, 1 = explicit points
    double q0_start;
    double q0_step;
    double q3_start;
    double q3_step;
    double q0_min;
    double q0_max;
    double q3_min;
    double q3_max;

    const double* d_q0_points;
    const double* d_q3_points;
    const GpuHadronTensorEntry* d_entries;
};

/// Kinematics query for batch differential cross-section evaluation
struct MecQuery {
    int probe_pdg;
    double E_probe;
    double m_probe;
    double Tl;
    double cos_l;
    double ml;
    double Q_value;
};

/// Candidate acceptance result from GPU rejection sampling
struct MecKinematicsCandidate {
    double Tl;
    double cos_l;
    double xsec;
    double xsec_pn;
    double xsec_delta;
    double xsec_delta_pn;
    bool accepted;
};

class GpuHadronTensor {
public:
    GpuHadronTensor(int device_id = 0);
    ~GpuHadronTensor();

    // Each tensor exclusively owns its device allocations.
    GpuHadronTensor(const GpuHadronTensor&) = delete;
    GpuHadronTensor& operator=(const GpuHadronTensor&) = delete;

    /// Load a table; success invalidates GPU data, failure preserves the table.
    bool LoadFile(const std::string& filepath);

    /// Replace the table and invalidate GPU data; invalid input returns false.
    bool Ingest(int Z, int A,
                const std::vector<double>& q0_vec,
                const std::vector<double>& q3_vec,
                const std::vector<GpuHadronTensorEntry>& entries);

    /// Upload table to GPU memory
    bool UploadToDevice();

    /// Free GPU allocations
    void FreeDevice();

    /// Host-side evaluation using CPU bilinear interpolation
    GpuHadronTensorEntry InterpolateHost(double q0, double q3) const;

    /// Host-side differential cross-section calculation
    double DiffXSecHost(int probe_pdg, double E_probe, double m_probe,
                        double Tl, double cos_l, double ml, double Q_value,
                        bool use_rosenbluth = false, double Vud = 0.9742) const;

    /// Host-input batch: uses current CPU data until UploadToDevice succeeds.
    void EvalDiffXSecBatch(const MecQuery* queries, double* out_xsec, int num_queries,
                           bool use_rosenbluth = false, double Vud = 0.9742) const;

    /// Device-input evaluation; throws if the current table is not uploaded.
    void EvalDiffXSecDevice(const MecQuery* d_queries, double* d_out_xsec, int num_queries,
                            bool use_rosenbluth = false, double Vud = 0.9742) const;

    /// Estimate the maximum on a grid for diagnostics; NOT a rejection bound.
    double FindMaxXSecGPU(int probe_pdg, double E_probe, double m_probe, double ml,
                          double T_min, double T_max, double costh_min, double costh_max,
                          double Q_value, bool use_rosenbluth = false, double Vud = 0.9742,
                          double safety_factor = 1.25) const;

    /// Deterministic envelope estimate on tensor nodes and cell midpoints,
    /// including projected phase-space edges. Not a certified analytic bound:
    /// keep the runtime bound check in SampleKinematicsGPU enabled.
    double FindMaxXSecOnTableGPU(int probe_pdg, double E_probe, double m_probe, double ml,
                                double T_min, double T_max, double costh_min, double costh_max,
                                double Q_value, bool use_rosenbluth, double Vud,
                                double Q3Max, double Q2min, double safety_factor = 2.0) const;

    /// Sample with a caller-supplied upper bound on the raw tensor cross section.
    /// Throws if the bound is invalid or any proposal exceeds it; no biased
    /// candidate is returned. FindMaxXSecGPU is not a certified bound.
    bool SampleKinematicsGPU(int probe_pdg, double E_probe, double m_probe, double ml,
                             double T_min, double T_max, double costh_min, double costh_max,
                             double Q_value, double xsec_max, bool use_rosenbluth,
                             double Vud, unsigned long long seed,
                             MecKinematicsCandidate& out_result, int batch_size = 4096,
                             double Q3Max = 1.e100, double Q2min = 0.0) const;

    /// Fixed-grid quadrature for diagnostics; does not certify convergence.
    /// GENIE's IntegrateGPU adapter adds refinement and CPU fallback.
    /// Each axis needs >=3 points; even counts round up to odd. Invalid or
    /// overflowing grids return false with zero output, as do nonfinite results.
    bool Integrate2DGPU(int probe_pdg, double E_probe, double m_probe, double ml,
                        double Delta_Q_value,
                        double T_min, double T_max, double costh_min, double costh_max,
                        double Q3Max, double Q2min,
                        bool use_rosenbluth, double Vud,
                        double& out_total_xsec,
                        int n_points_T = 257, int n_points_costh = 257) const;

    // Static GPU status & control
    static bool IsGpuAvailable();
    static bool IsGpuEnabled();
    static void SetGpuEnabled(bool enabled);
    static void SetDefaultDevice(int device_id);
    static int DefaultDevice();

    // Getters
    inline int Z() const { return fHeader.Z; }
    inline int A() const { return fHeader.A; }
    inline int TargetPdg() const { return fHeader.target_pdg; }
    inline int NumQ0() const { return fHeader.num_q0; }
    inline int NumQ3() const { return fHeader.num_q3; }
    inline double Q0Min() const { return fHeader.q0_min; }
    inline double Q0Max() const { return fHeader.q0_max; }
    inline double Q3Min() const { return fHeader.q3_min; }
    inline double Q3Max() const { return fHeader.q3_max; }
    inline bool IsOnDevice() const { return fIsOnDevice; }
    inline const GpuHadronTensorData& DeviceData() const { return fHeader; }

private:
    int fDeviceId;
    bool fIsLoaded;
    bool fIsOnDevice;

    GpuHadronTensorData fHeader;

    std::vector<double> fHostQ0;
    std::vector<double> fHostQ3;
    std::vector<GpuHadronTensorEntry> fHostEntries;

    // Device memory pointers
    double* d_q0;
    double* d_q3;
    GpuHadronTensorEntry* d_entries;

    // Pre-allocated device buffers for rejection sampling & reduction
    MecKinematicsCandidate* d_sample_candidates;
    int* d_sample_count;
    double* d_max_xsec;
    double* d_integral;
    int fSampleBatchSize;
};

} // namespace gpuspline

#endif // GPU_HADRON_TENSOR_H
