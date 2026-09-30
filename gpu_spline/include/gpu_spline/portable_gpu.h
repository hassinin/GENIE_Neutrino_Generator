#ifndef GPU_SPLINE_PORTABLE_GPU_H
#define GPU_SPLINE_PORTABLE_GPU_H

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

#if defined(__HIP__) || defined(__HIPCC__) || defined(USE_HIP)
    #define GPU_ENABLE_HIP 1
    #if defined(__HIP__)
        // hipcc/clang compiling HIP code (host and device passes)
        #include <hip/hip_runtime.h>
        #define GPU_HOST __host__
        #define GPU_DEVICE __device__
        #define GPU_HOST_DEVICE __host__ __device__
        #define GPU_GLOBAL __global__
        #define GPU_SHARED __shared__
        #define GPU_CONSTANT __constant__
    #else
        // Host-only C++ built with g++ -DUSE_HIP -D__HIP_PLATFORM_AMD__:
        // the plain C++ runtime API is enough.
        #include <hip/hip_runtime_api.h>
        #define GPU_HOST
        #define GPU_DEVICE
        #define GPU_HOST_DEVICE
        #define GPU_GLOBAL
        #define GPU_SHARED
        #define GPU_CONSTANT
    #endif

    #define gpuMalloc hipMalloc
    #define gpuFree hipFree
    #define gpuMemcpy hipMemcpy
    #define gpuMemset hipMemset
    #define gpuSetDevice hipSetDevice
    #define gpuGetDeviceCount hipGetDeviceCount
    #define gpuMemcpyHostToDevice hipMemcpyHostToDevice
    #define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
    #define gpuDeviceSynchronize hipDeviceSynchronize
    #define gpuGetLastError hipGetLastError
    #define gpuGetErrorString hipGetErrorString
    #define gpuSuccess hipSuccess
    typedef hipError_t gpuError_t;
    typedef hipStream_t gpuStream_t;

#elif defined(__CUDACC__) || defined(__NVCC__) || !defined(CPU_ONLY)
    #define GPU_ENABLE_CUDA 1
    // NVIDIA CUDA compilation (both nvcc device and g++ host)
    #include <cuda_runtime.h>

    #if defined(__CUDACC__) || defined(__NVCC__)
        #define GPU_HOST __host__
        #define GPU_DEVICE __device__
        #define GPU_HOST_DEVICE __host__ __device__
        #define GPU_GLOBAL __global__
        #define GPU_SHARED __shared__
        #define GPU_CONSTANT __constant__
    #else
        #define GPU_HOST
        #define GPU_DEVICE
        #define GPU_HOST_DEVICE
        #define GPU_GLOBAL
        #define GPU_SHARED
        #define GPU_CONSTANT
    #endif

    #define gpuMalloc cudaMalloc
    #define gpuFree cudaFree
    #define gpuMemcpy cudaMemcpy
    #define gpuMemset cudaMemset
    #define gpuSetDevice cudaSetDevice
    #define gpuGetDeviceCount cudaGetDeviceCount
    #define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
    #define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
    #define gpuDeviceSynchronize cudaDeviceSynchronize
    #define gpuGetLastError cudaGetLastError
    #define gpuGetErrorString cudaGetErrorString
    #define gpuSuccess cudaSuccess
    typedef cudaError_t gpuError_t;
    typedef cudaStream_t gpuStream_t;

#else
    // Pure CPU fallback (no CUDA/HIP runtime installed)
    #define GPU_HOST
    #define GPU_DEVICE
    #define GPU_HOST_DEVICE
    #define GPU_GLOBAL
    #define GPU_SHARED
    #define GPU_CONSTANT

    typedef int gpuError_t;
    typedef void* gpuStream_t;
    #define gpuSuccess 0
    #define gpuMemcpyHostToDevice 1
    #define gpuMemcpyDeviceToHost 2
    inline const char* gpuGetErrorString(gpuError_t) { return "CPU fallback (no error)"; }
    inline gpuError_t gpuGetLastError() { return 0; }
    inline gpuError_t gpuDeviceSynchronize() { return 0; }
    inline gpuError_t gpuSetDevice(int) { return 0; }
    inline gpuError_t gpuMemset(void* ptr, int value, size_t size) { memset(ptr, value, size); return 0; }
    inline gpuError_t gpuMalloc(void** ptr, size_t size) { *ptr = malloc(size); return 0; }
    inline gpuError_t gpuFree(void* ptr) { free(ptr); return 0; }
    inline gpuError_t gpuMemcpy(void* dst, const void* src, size_t size, int) { memcpy(dst, src, size); return 0; }
#endif

#include "gpu_spline/gpu_status.h"

// Status reported when no runtime code applies (the value of cudaErrorUnknown).
#define GPU_SPLINE_UNKNOWN_ERROR 999

// Portable error checks
// Initialization uses this nonfatal check so callers can fall back to CPU.
#define gpuTry(ans) gpuTryAt((ans), __FILE__)
inline bool gpuTryAt(gpuError_t code, const char* file) {
    const bool injected = gpu_spline::CheckFaultInjection(file);
    if (code == gpuSuccess && !injected) return true;
    fprintf(stderr, "GPU initialization error: %s\n",
            code != gpuSuccess ? gpuGetErrorString(code) : "injected fault");
    // Do not leave a handled initialization failure as the next launch's error.
    (void)gpuGetLastError();
    return false;
}

// Per-call paths throw gpu_spline::GpuDeviceError so callers can fall back to
// the CPU instead of the process exiting.
#define GPU_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(gpuError_t code, const char *file, int line) {
    const bool injected = gpu_spline::CheckFaultInjection(file);
    if (code == gpuSuccess && !injected) return;
    if (code != gpuSuccess) {
        (void)gpuGetLastError();  // clear a non-sticky error before the caller continues
        gpu_spline::ReportDeviceError(static_cast<int>(code), gpuGetErrorString(code), file, line);
    }
    gpu_spline::ReportDeviceError(GPU_SPLINE_UNKNOWN_ERROR, "injected fault (GPU_SPLINE_FAIL_AFTER)", file, line);
}

// Owns a device allocation; frees it without error checks so it is safe while
// an exception unwinds.
template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t n) {
        if (n == 0) return;
        void* p = nullptr;
        const gpuError_t code = gpuMalloc(&p, n * sizeof(T));
        if (code == gpuSuccess) ptr_ = static_cast<T*>(p);
        try {
            GPU_CHECK(code);
        } catch (...) {
            if (ptr_) (void)gpuFree(ptr_);
            ptr_ = nullptr;
            throw;
        }
    }
    ~DeviceBuffer() { if (ptr_) (void)gpuFree(ptr_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return ptr_; }
private:
    T* ptr_ = nullptr;
};

// Portable device math functions
GPU_HOST_DEVICE inline double gpu_max(double a, double b) {
    return (a > b) ? a : b;
}

GPU_HOST_DEVICE inline double gpu_min(double a, double b) {
    return (a < b) ? a : b;
}

GPU_HOST_DEVICE inline bool gpu_isnan(double x) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return isnan(x);
#else
    return std::isnan(x);
#endif
}

#endif // GPU_SPLINE_PORTABLE_GPU_H
