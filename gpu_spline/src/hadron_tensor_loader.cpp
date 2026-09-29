#include "gpu_spline/gpu_hadron_tensor.h"
#include "gpu_spline/tensor_kinematics.h"
#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <limits>

namespace gpuspline {

namespace {
    inline double real_sqrt(double x) {
        return (x <= 0.0) ? 0.0 : std::sqrt(x);
    }
}

GpuHadronTensor::GpuHadronTensor(int device_id)
    : fDeviceId(device_id), fIsLoaded(false), fIsOnDevice(false),
      d_q0(nullptr), d_q3(nullptr), d_entries(nullptr),
      d_sample_candidates(nullptr), d_sample_count(nullptr), d_max_xsec(nullptr), d_integral(nullptr),
      fSampleBatchSize(4096)
{
    fHeader.Z = 0;
    fHeader.A = 0;
    fHeader.target_pdg = 0;
    fHeader.num_q0 = 0;
    fHeader.num_q3 = 0;
    fHeader.q0_flag = 0;
    fHeader.q3_flag = 0;
    fHeader.q0_start = 0.0;
    fHeader.q0_step = 0.0;
    fHeader.q3_start = 0.0;
    fHeader.q3_step = 0.0;
    fHeader.q0_min = 0.0;
    fHeader.q0_max = 0.0;
    fHeader.q3_min = 0.0;
    fHeader.q3_max = 0.0;
    fHeader.d_q0_points = nullptr;
    fHeader.d_q3_points = nullptr;
    fHeader.d_entries = nullptr;
}

GpuHadronTensor::~GpuHadronTensor() {
    FreeDevice();
}

bool GpuHadronTensor::LoadFile(const std::string& filepath) {
    std::ifstream in_file(filepath.c_str());
    if (!in_file.is_open()) {
        std::cerr << "[GpuHadronTensor] Cannot open file: " << filepath << std::endl;
        return false;
    }

    // Skip comment line
    std::string dummy;
    std::getline(in_file, dummy);

    std::string type_name;
    int Z = 0, A = 0, num_q0 = 0, num_q3 = 0;
    if (!(in_file >> Z >> A >> type_name >> num_q0 >> num_q3)) {
        std::cerr << "[GpuHadronTensor] Failed reading header line from " << filepath << std::endl;
        return false;
    }

    if (num_q0 < 2 || num_q3 < 2 ||
        num_q0 > std::numeric_limits<int>::max() / num_q3) return false;

    // Parse into temporary storage so a failed reload preserves the old table.
    std::vector<double> q0, q3;
    std::vector<GpuHadronTensorEntry> entries;
    auto read_grid = [&in_file](int count, int& flag, double& start,
                                double& step, std::vector<double>& grid) {
        if (!(in_file >> flag) || (flag != 0 && flag != 1)) return false;
        if (flag == 0 && !(in_file >> start >> step)) return false;
        for (int k = 0; k < count; ++k) {
            double value = start + k * step;
            if (flag == 1 && !(in_file >> value)) return false;
            grid.push_back(value);
        }
        return true;
    };
    int q0_flag = 0, q3_flag = 0;
    double q0_start = 0., q0_step = 0., q3_start = 0., q3_step = 0.;
    if (!read_grid(num_q0, q0_flag, q0_start, q0_step, q0) ||
        !read_grid(num_q3, q3_flag, q3_start, q3_step, q3)) return false;

    for (int j = 0; j < num_q0; ++j) {
        for (int k = 0; k < num_q3; ++k) {
            double w00, rew0z, wxx, imwxy, wzz;
            if (!(in_file >> w00 >> rew0z >> wxx >> imwxy >> wzz)) {
                std::cerr << "[GpuHadronTensor] Error reading table data at (" << j << "," << k << ")" << std::endl;
                return false;
            }
            entries.emplace_back(w00, rew0z, wxx, imwxy, wzz);
        }
    }

    if (!Ingest(Z, A, q0, q3, entries)) return false;
    fHeader.q0_flag = q0_flag;
    fHeader.q3_flag = q3_flag;
    fHeader.q0_start = q0_start;
    fHeader.q0_step = q0_step;
    fHeader.q3_start = q3_start;
    fHeader.q3_step = q3_step;

    std::cout << "[GpuHadronTensor] Loaded tensor " << type_name << " for target " 
              << fHeader.target_pdg << " (Z=" << Z << ", A=" << A << ") grid: " 
              << num_q0 << "x" << num_q3 << " points from " << filepath << std::endl;
    return true;
}

bool GpuHadronTensor::Ingest(int Z, int A,
                            const std::vector<double>& q0_vec,
                            const std::vector<double>& q3_vec,
                            const std::vector<GpuHadronTensorEntry>& entries)
{
    auto valid_grid = [](const std::vector<double>& grid) {
        if (grid.size() < 2) return false;
        for (size_t i = 0; i < grid.size(); ++i) {
            if (!std::isfinite(grid[i]) || (i && grid[i] <= grid[i-1])) return false;
        }
        return true;
    };
    if (!valid_grid(q0_vec) || !valid_grid(q3_vec) ||
        q0_vec.size() > static_cast<size_t>(std::numeric_limits<int>::max()) / q3_vec.size() ||
        entries.size() != q0_vec.size() * q3_vec.size()) return false;
    for (const auto& entry : entries) {
        if (!std::isfinite(entry.W00) || !std::isfinite(entry.ReW0z) ||
            !std::isfinite(entry.Wxx) || !std::isfinite(entry.ImWxy) ||
            !std::isfinite(entry.Wzz)) return false;
    }

    // Invalidate both metadata pointers and allocations before changing shape.
    FreeDevice();
    fHostQ0 = q0_vec;
    fHostQ3 = q3_vec;
    fHostEntries = entries;

    fHeader.Z = Z;
    fHeader.A = A;
    fHeader.target_pdg = 1000000000 + 10000 * Z + 10 * A;
    fHeader.num_q0 = static_cast<int>(q0_vec.size());
    fHeader.num_q3 = static_cast<int>(q3_vec.size());
    fHeader.q0_flag = 1; // Explicit points by default
    fHeader.q3_flag = 1;
    fHeader.q0_start = q0_vec.front();
    fHeader.q0_step = (q0_vec.size() > 1) ? (q0_vec[1] - q0_vec[0]) : 0.0;
    fHeader.q3_start = q3_vec.front();
    fHeader.q3_step = (q3_vec.size() > 1) ? (q3_vec[1] - q3_vec[0]) : 0.0;
    fHeader.q0_min = q0_vec.front();
    fHeader.q0_max = q0_vec.back();
    fHeader.q3_min = q3_vec.front();
    fHeader.q3_max = q3_vec.back();

    fIsLoaded = true;
    return true;
}

bool GpuHadronTensor::UploadToDevice() {
    if (!fIsLoaded) {
        std::cerr << "[GpuHadronTensor] Cannot upload: no table loaded." << std::endl;
        return false;
    }

    FreeDevice();

#if defined(GPU_ENABLE_CUDA) || defined(GPU_ENABLE_HIP)
    if (!gpuTry(gpuSetDevice(fDeviceId))) return false;

    size_t q0_bytes = fHeader.num_q0 * sizeof(double);
    size_t q3_bytes = fHeader.num_q3 * sizeof(double);
    size_t entries_bytes = fHeader.num_q0 * fHeader.num_q3 * sizeof(GpuHadronTensorEntry);

    if (!gpuTry(gpuMalloc((void**)&d_q0, q0_bytes))) { FreeDevice(); return false; }
    if (!gpuTry(gpuMalloc((void**)&d_q3, q3_bytes))) { FreeDevice(); return false; }
    if (!gpuTry(gpuMalloc((void**)&d_entries, entries_bytes))) { FreeDevice(); return false; }

    if (!gpuTry(gpuMemcpy(d_q0, fHostQ0.data(), q0_bytes, gpuMemcpyHostToDevice))) { FreeDevice(); return false; }
    if (!gpuTry(gpuMemcpy(d_q3, fHostQ3.data(), q3_bytes, gpuMemcpyHostToDevice))) { FreeDevice(); return false; }
    if (!gpuTry(gpuMemcpy(d_entries, fHostEntries.data(), entries_bytes, gpuMemcpyHostToDevice))) { FreeDevice(); return false; }

    // Preallocate sampling buffers
    fSampleBatchSize = 4096;
    if (!gpuTry(gpuMalloc((void**)&d_sample_candidates, fSampleBatchSize * sizeof(MecKinematicsCandidate)))) { FreeDevice(); return false; }
    if (!gpuTry(gpuMalloc((void**)&d_sample_count, sizeof(int)))) { FreeDevice(); return false; }
    if (!gpuTry(gpuMalloc((void**)&d_max_xsec, sizeof(double)))) { FreeDevice(); return false; }
    if (!gpuTry(gpuMalloc((void**)&d_integral, sizeof(double)))) { FreeDevice(); return false; }

    fHeader.d_q0_points = d_q0;
    fHeader.d_q3_points = d_q3;
    fHeader.d_entries = d_entries;
    fIsOnDevice = true;

    std::cout << "[GpuHadronTensor] Uploaded " << (entries_bytes + q0_bytes + q3_bytes) / 1024.0 
              << " KB to GPU " << fDeviceId << std::endl;
    return true;
#else
    // Host-only fallback
    fHeader.d_q0_points = fHostQ0.data();
    fHeader.d_q3_points = fHostQ3.data();
    fHeader.d_entries = fHostEntries.data();
    fIsOnDevice = true;
    return true;
#endif
}

void GpuHadronTensor::FreeDevice() {
#if defined(GPU_ENABLE_CUDA) || defined(GPU_ENABLE_HIP)
    if (d_q0 || d_q3 || d_entries) gpuSetDevice(fDeviceId);
    if (d_q0) { gpuFree(d_q0); d_q0 = nullptr; }
    if (d_q3) { gpuFree(d_q3); d_q3 = nullptr; }
    if (d_entries) { gpuFree(d_entries); d_entries = nullptr; }
    if (d_sample_candidates) { gpuFree(d_sample_candidates); d_sample_candidates = nullptr; }
    if (d_sample_count) { gpuFree(d_sample_count); d_sample_count = nullptr; }
    if (d_max_xsec) { gpuFree(d_max_xsec); d_max_xsec = nullptr; }
    if (d_integral) { gpuFree(d_integral); d_integral = nullptr; }
#endif
    fHeader.d_q0_points = nullptr;
    fHeader.d_q3_points = nullptr;
    fHeader.d_entries = nullptr;
    fIsOnDevice = false;
}

namespace {
    static bool g_gpu_hadron_tensor_enabled = false;
    static int g_gpu_hadron_tensor_device = 0;
}

bool GpuHadronTensor::IsGpuAvailable() {
#if defined(GPU_ENABLE_CUDA) || defined(GPU_ENABLE_HIP)
    int count = 0;
    gpuError_t err = gpuGetDeviceCount(&count);
    return (err == gpuSuccess && DefaultDevice() >= 0 && DefaultDevice() < count);
#else
    return false;
#endif
}

void GpuHadronTensor::SetDefaultDevice(int device_id) {
    g_gpu_hadron_tensor_device = device_id;
}

int GpuHadronTensor::DefaultDevice() {
    return g_gpu_hadron_tensor_device;
}

bool GpuHadronTensor::IsGpuEnabled() {
    if (!IsGpuAvailable()) return false;
    if (g_gpu_hadron_tensor_enabled) return true;
    const char* env = std::getenv("GENIE_USE_GPU");
    if (env && std::string(env) != "0") return true;
    return false;
}

void GpuHadronTensor::SetGpuEnabled(bool enabled) {
    g_gpu_hadron_tensor_enabled = enabled;
}

GpuHadronTensorEntry GpuHadronTensor::InterpolateHost(double q0, double q3) const {
    if (!fIsLoaded) return GpuHadronTensorEntry();

    double eval_q0 = std::max(fHeader.q0_min, std::min(q0, fHeader.q0_max));
    double eval_q3 = std::max(fHeader.q3_min, std::min(q3, fHeader.q3_max));

    int ix_lo = 0, ix_hi = 0;
    int iy_lo = 0, iy_hi = 0;

    if (fHeader.q0_flag == 0 && fHeader.q0_step > 0.0) {
        ix_lo = static_cast<int>((eval_q0 - fHeader.q0_start) / fHeader.q0_step);
        ix_lo = std::max(0, std::min(ix_lo, fHeader.num_q0 - 2));
        ix_hi = ix_lo + 1;
    } else {
        auto it = std::lower_bound(fHostQ0.begin(), fHostQ0.end(), eval_q0);
        ix_lo = static_cast<int>(std::distance(fHostQ0.begin(), it));
        if (ix_lo >= fHeader.num_q0 - 1) ix_lo = fHeader.num_q0 - 2;
        if (ix_lo > 0 && fHostQ0[ix_lo] > eval_q0) ix_lo--;
        ix_hi = ix_lo + 1;
    }

    if (fHeader.q3_flag == 0 && fHeader.q3_step > 0.0) {
        iy_lo = static_cast<int>((eval_q3 - fHeader.q3_start) / fHeader.q3_step);
        iy_lo = std::max(0, std::min(iy_lo, fHeader.num_q3 - 2));
        iy_hi = iy_lo + 1;
    } else {
        auto it = std::lower_bound(fHostQ3.begin(), fHostQ3.end(), eval_q3);
        iy_lo = static_cast<int>(std::distance(fHostQ3.begin(), it));
        if (iy_lo >= fHeader.num_q3 - 1) iy_lo = fHeader.num_q3 - 2;
        if (iy_lo > 0 && fHostQ3[iy_lo] > eval_q3) iy_lo--;
        iy_hi = iy_lo + 1;
    }

    double x1 = fHostQ0[ix_lo], x2 = fHostQ0[ix_hi];
    double y1 = fHostQ3[iy_lo], y2 = fHostQ3[iy_hi];

    const GpuHadronTensorEntry& z11 = fHostEntries[ix_lo * fHeader.num_q3 + iy_lo];
    const GpuHadronTensorEntry& z21 = fHostEntries[ix_hi * fHeader.num_q3 + iy_lo];
    const GpuHadronTensorEntry& z12 = fHostEntries[ix_lo * fHeader.num_q3 + iy_hi];
    const GpuHadronTensorEntry& z22 = fHostEntries[ix_hi * fHeader.num_q3 + iy_hi];

    double dy = y2 - y1;
    double dx = x2 - x1;
    double wy1 = (dy > 0.0) ? (y2 - eval_q3) / dy : 1.0;
    double wy2 = (dy > 0.0) ? (eval_q3 - y1) / dy : 0.0;
    double wx1 = (dx > 0.0) ? (x2 - eval_q0) / dx : 1.0;
    double wx2 = (dx > 0.0) ? (eval_q0 - x1) / dx : 0.0;

    GpuHadronTensorEntry z1 = z11 * wy1 + z12 * wy2;
    GpuHadronTensorEntry z2 = z21 * wy1 + z22 * wy2;
    return z1 * wx1 + z2 * wx2;
}

double GpuHadronTensor::DiffXSecHost(int probe_pdg, double E_probe, double m_probe,
                                    double Tl, double cos_l, double ml, double Q_value,
                                    bool use_rosenbluth, double Vud) const
{
    double El = Tl + ml;
    double q0 = E_probe - El;
    double q0_corrected = q0 - Q_value;

    double k_initial = real_sqrt(E_probe * E_probe - m_probe * m_probe);
    double k_final = real_sqrt(Tl * Tl + 2.0 * ml * Tl);
    double q_mag2 = TensorQ3Squared(k_initial, k_final, cos_l);
    double q_mag = real_sqrt(q_mag2);

    if (!(q_mag2 > 0.0) || !TensorInSupport(q0_corrected, q_mag, E_probe,
            fHeader.q0_min, fHeader.q0_max, fHeader.q3_min, fHeader.q3_max)) {
        return 0.0;
    }

    double q2 = q0 * q0 - q_mag2;
    double Q2 = q_mag2 - q0 * q0;
    if (Q2 < 0.0) return 0.0;

    GpuHadronTensorEntry entry = InterpolateHost(q0_corrected, q_mag);

    const double kPi = 3.14159265358979323846;
    const double kGF = 1.16639E-5;
    const double kGF2 = kGF * kGF;

    double s2_half = (1.0 - cos_l) * 0.5;
    double c2_half = 1.0 - s2_half;

    if (probe_pdg == 11) {
        // GENIE has distinct electron contractions. Only Rosenbluth applies
        // the finite-mass Mott factor and the tensor Wxx / 2 correction.
        if (Q2 <= 0.0) return 0.0;
        const double alpha = 1.0 / 137.03599976;
        const double mott_amp = use_rosenbluth ? 2.0 * El * alpha / q2 :
                                               alpha / (2.0 * E_probe * s2_half);
        const double longitudinal = (q2 / q_mag2) * (q2 / q_mag2) * entry.W00 * c2_half;
        const double transverse = (2.0 * s2_half - q2 * c2_half / q_mag2) * entry.Wxx;
        return 2.0 * kPi * mott_amp * mott_amp *
            (longitudinal + (use_rosenbluth ? transverse / 2.0 : transverse));
    }
    const int abs_pdg = (probe_pdg < 0) ? -probe_pdg : probe_pdg;
    if (abs_pdg != 12 && abs_pdg != 14 && abs_pdg != 16) return 0.0;

    if (!use_rosenbluth) {
        // Standard Valencia formulation
        double w1 = entry.Wxx * 0.5;
        double temp_1 = (std::pow(q0, 2) / q_mag2) * (entry.Wzz - entry.Wxx);
        double temp_2 = 2.0 * q0 * entry.ReW0z / q_mag;
        double w2 = (entry.W00 + entry.Wxx + temp_1 - temp_2) * 0.5;

        double w3 = entry.ImWxy / q_mag;
        if (probe_pdg < 0) w3 *= -1.0;

        double w4 = (entry.Wzz - entry.Wxx) / (2.0 * q_mag2);
        double w5 = (entry.ReW0z - q0 * (entry.Wzz - entry.Wxx) / q_mag) / q_mag;

        double part_1 = (2.0 * w1 * s2_half) + (w2 * c2_half) - (w3 * (E_probe + El) * s2_half);
        double part_2 = (w1 * cos_l) - (w2 * 0.5 * cos_l)
                      + (w3 * 0.5 * (El + k_final - (E_probe + El) * cos_l))
                      + (w4 * 0.5 * (ml * ml * cos_l + 2.0 * El * (El + k_final) * s2_half))
                      - (w5 * 0.5 * (El + k_final));

        double denom = El * (El + k_final);
        double all_terms = part_1 + (denom > 0.0 ? (ml * ml * part_2 / denom) : 0.0);

        return (2.0 / kPi) * k_final * El * kGF2 * all_terms;
    } else {
        // SuSAv2 Rosenbluth formulation
        double v0 = 4.0 * El * E_probe + q2;
        double Vud2 = Vud * Vud;
        double sig0 = kGF2 * Vud2 * v0 * k_final / (4.0 * kPi * E_probe);

        double neg_q2 = -q2;
        if (neg_q2 <= 0.0) return 0.0;
        double sqrt_neg_q2 = std::sqrt(neg_q2);

        double xdelta = ml / sqrt_neg_q2;
        double xrho = neg_q2 / q_mag2;
        double xrhop = q_mag / (El + E_probe);
        double tan2th2 = neg_q2 / v0;

        double xdelta2 = xdelta * xdelta;
        double VCC = 1.0 - xdelta2 * tan2th2;
        double VCL = q0 / q_mag + tan2th2 * xdelta2 / xrhop;
        double VLL = (q0 * q0) / q_mag2 + xdelta2 * tan2th2 * (1.0 + 2.0 * q0 / q_mag / xrhop + xrho * xdelta2);
        double VT = xrho * 0.5 + tan2th2 - tan2th2 * xdelta2 / xrhop * (q0 / q_mag + xdelta2 * xrho * xrhop * 0.5);
        double VTP = tan2th2 / xrhop * (1.0 - q0 / q_mag * xrhop * xdelta2);

        double RCC = entry.W00;
        double RCL = -entry.ReW0z;
        double RLL = entry.Wzz;
        double RT = 2.0 * entry.Wxx;
        double RTP = -entry.ImWxy;
        if (probe_pdg < 0) RTP *= -1.0;

        double xsec = sig0 * (VCC * RCC + 2.0 * VCL * RCL + VLL * RLL + VT * RT + 2.0 * VTP * RTP);
        return (xsec < 0.0) ? 0.0 : xsec;
    }
}

} // namespace gpuspline
