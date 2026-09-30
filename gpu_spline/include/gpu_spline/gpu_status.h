#ifndef GPU_SPLINE_GPU_STATUS_H
#define GPU_SPLINE_GPU_STATUS_H

#include <stdexcept>
#include <string>

namespace gpu_spline {

// Thrown by per-call GPU paths when the CUDA/HIP runtime reports an error.
// Physics and configuration guards (for example an exceeded sampling bound)
// throw other std::runtime_error types, so callers can fall back to the CPU
// for device failures only.
class GpuDeviceError : public std::runtime_error {
public:
    GpuDeviceError(int code, const std::string& what);
    ~GpuDeviceError() override;
    int code() const { return code_; }
private:
    int code_;
};

// Throws GpuDeviceError with the runtime message and source location.
[[noreturn]] void ReportDeviceError(int code, const char* message, const char* file, int line);

// Process-wide switch. The library never sets it; callers such as GENIE call
// DisableGpu after the first device error. DisableGpu returns true only for
// the call that disabled the GPU, so exactly one caller reports the failure.
bool GpuDisabled();
bool DisableGpu(const std::string& reason);
std::string GpuDisableReason();

// Fault injection for tests. Once armed, the n-th device check that matches
// file_filter fails, and every check after it fails too (like a sticky device
// fault). n = 0 disarms. Production runs can arm it with the environment
// variables GPU_SPLINE_FAIL_AFTER=n and GPU_SPLINE_FAIL_SITE=<file substring>.
void SetFaultInjection(long checks_from_now, const char* file_filter = nullptr);
// Called by every device check; true means treat this check as failed.
bool CheckFaultInjection(const char* file);

} // namespace gpu_spline

#endif // GPU_SPLINE_GPU_STATUS_H
