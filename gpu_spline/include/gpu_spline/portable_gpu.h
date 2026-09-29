#ifndef GPU_SPLINE_PORTABLE_GPU_H
#define GPU_SPLINE_PORTABLE_GPU_H

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

#if defined(__HIPCC__) || defined(__HIP__) || defined(USE_HIP)
    #define GPU_ENABLE_HIP 1
    // AMD ROCm / HIP compilation (both device and host)
    #include <hip/hip_runtime.h>
    #define GPU_HOST __host__
    #define GPU_DEVICE __device__
    #define GPU_HOST_DEVICE __host__ __device__
    #define GPU_GLOBAL __global__
    #define GPU_SHARED __shared__
    #define GPU_CONSTANT __constant__

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

// Portable error check macro
// Initialization uses this nonfatal check so callers can fall back to CPU.
inline bool gpuTry(gpuError_t code) {
    if (code == gpuSuccess) return true;
    fprintf(stderr, "GPU initialization error: %s\n", gpuGetErrorString(code));
    // Do not leave a handled initialization failure as the next launch's error.
    gpuGetLastError();
    return false;
}
#define GPU_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(gpuError_t code, const char *file, int line, bool abort=true) {
    if (code != gpuSuccess) {
        fprintf(stderr, "GPU Error: %s %s %d\n", gpuGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

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
