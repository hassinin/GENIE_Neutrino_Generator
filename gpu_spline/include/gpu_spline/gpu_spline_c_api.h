#ifndef GPU_SPLINE_C_API_H
#define GPU_SPLINE_C_API_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* gpu_spline_handle_t;

// Lifecycle
gpu_spline_handle_t gpu_spline_create();
void gpu_spline_destroy(gpu_spline_handle_t handle);

// Loading & Initialization
int gpu_spline_load_xml(gpu_spline_handle_t handle, const char* filepath);
int gpu_spline_init_gpu(gpu_spline_handle_t handle, int device_id);
void gpu_spline_free_gpu(gpu_spline_handle_t handle);

// Query Metadata
int gpu_spline_get_num_splines(gpu_spline_handle_t handle);
// Unqualified lookup returns -1 if missing, -2 if ambiguous across tunes.
int gpu_spline_find_id(gpu_spline_handle_t handle, const char* name);
int gpu_spline_find_id_for_tune(gpu_spline_handle_t handle, const char* name, const char* tune);
const char* gpu_spline_get_name(gpu_spline_handle_t handle, int spline_id);
// Untuned splines (and invalid IDs) return an empty string.
const char* gpu_spline_get_tune(gpu_spline_handle_t handle, int spline_id);

// Evaluation
double gpu_spline_eval_single_cpu(gpu_spline_handle_t handle, int spline_id, double energy);
void gpu_spline_eval_batch_cpu(gpu_spline_handle_t handle, const double* energies, const int* spline_ids, double* results, size_t n);
void gpu_spline_eval_batch_gpu(gpu_spline_handle_t handle, const double* energies, const int* spline_ids, double* results, size_t n);
void gpu_spline_eval_device(gpu_spline_handle_t handle, const double* d_energies, const int* d_spline_ids, double* d_results, size_t n, void* stream);
// Nonthrowing device-input submission: 1 on success (including n == 0), 0 on
// invalid arguments, missing device tables, or a device/launch error. Failure
// leaves results unusable. Success is asynchronous; synchronize the stream to
// detect execution errors before consuming results. The legacy void function
// delegates here and exposes failures through the same error accessor.
int gpu_spline_eval_device_checked(gpu_spline_handle_t handle, const double* d_energies, const int* d_spline_ids, double* d_results, size_t n, void* stream);
// Thread-local message for the last device evaluation; empty on success.
// Valid until the next device evaluation on the calling thread.
const char* gpu_spline_device_last_error(void);

#ifdef __cplusplus
}
#endif

#endif // GPU_SPLINE_C_API_H
