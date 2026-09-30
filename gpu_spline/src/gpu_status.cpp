#include "gpu_spline/gpu_status.h"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <sstream>

namespace gpu_spline {

GpuDeviceError::GpuDeviceError(int code, const std::string& what)
    : std::runtime_error(what), code_(code) {}

GpuDeviceError::~GpuDeviceError() = default;

[[noreturn]] void ReportDeviceError(int code, const char* message, const char* file, int line) {
    std::ostringstream what;
    what << "GPU runtime error: " << (message ? message : "unknown") << " (" << file << ":" << line << ")";
    throw GpuDeviceError(code, what.str());
}

namespace {
std::mutex g_mutex;
std::atomic<bool> g_disabled{false};
std::string g_reason;

// Fault injection. g_remaining counts matching checks until the fault fires;
// g_failing makes it sticky afterwards.
std::once_flag g_env_once;
std::atomic<long> g_remaining{0};
std::atomic<bool> g_failing{false};
std::string g_filter;

void ReadEnvironment() {
    const char* after = std::getenv("GPU_SPLINE_FAIL_AFTER");
    if (!after || !*after) return;
    const long n = std::strtol(after, nullptr, 10);
    if (n <= 0) return;
    const char* site = std::getenv("GPU_SPLINE_FAIL_SITE");
    std::lock_guard<std::mutex> lock(g_mutex);
    g_filter = site ? site : "";
    g_remaining = n;
}
} // namespace

bool GpuDisabled() { return g_disabled.load(); }

bool DisableGpu(const std::string& reason) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_disabled.load()) return false;
    g_reason = reason;
    g_disabled = true;
    return true;
}

std::string GpuDisableReason() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_reason;
}

void SetFaultInjection(long checks_from_now, const char* file_filter) {
    std::call_once(g_env_once, [] {});  // explicit arming overrides the environment
    std::lock_guard<std::mutex> lock(g_mutex);
    g_failing = false;
    g_filter = file_filter ? file_filter : "";
    g_remaining = checks_from_now > 0 ? checks_from_now : 0;
}

bool CheckFaultInjection(const char* file) {
    std::call_once(g_env_once, ReadEnvironment);
    if (g_failing.load()) return true;
    if (g_remaining.load() <= 0) return false;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (!g_filter.empty() && (!file || !std::strstr(file, g_filter.c_str()))) return false;
    }
    if (g_remaining.fetch_sub(1) == 1) {
        g_failing = true;
        return true;
    }
    return false;
}

} // namespace gpu_spline
