#ifndef GPU_SPLINE_PORTABLE_RAND_H
#define GPU_SPLINE_PORTABLE_RAND_H

// Device-side Philox4x32-10 random numbers: cuRAND on NVIDIA, hipRAND on AMD.
// Include only from device code (.cu files). The two libraries implement the
// same generator, but AMD and NVIDIA sequences are not verified to be identical.

#include "portable_gpu.h"

#if defined(GPU_ENABLE_HIP)
    #include <hiprand/hiprand_kernel.h>
    typedef hiprandStatePhilox4_32_10_t gpuRandPhiloxState;
    #define gpu_rand_init hiprand_init
    #define gpu_rand_uniform4 hiprand_uniform4
#elif defined(GPU_ENABLE_CUDA)
    #include <curand_kernel.h>
    typedef curandStatePhilox4_32_10_t gpuRandPhiloxState;
    #define gpu_rand_init curand_init
    #define gpu_rand_uniform4 curand_uniform4
#endif

#endif // GPU_SPLINE_PORTABLE_RAND_H
