#ifndef GPU_SPLINE_GPU_QEL_INTEGRATOR_H
#define GPU_SPLINE_GPU_QEL_INTEGRATOR_H
#include "qel_integrand.h"
#include <vector>
#include <string>
namespace gpu_spline {
struct QelIntegralResult { double value=0, error=0; bool converged=false; unsigned evaluations=0; };
// Returns false on device failure. The caller must evaluate the SAME saved
// nuclear samples on CPU, never draw replacement nucleons on fallback.
bool IntegrateQelGpu(const QelParameters&, const std::vector<QelSample>&,
                     double relative_tolerance, unsigned max_evaluations, int device,
                     std::vector<QelIntegralResult>&, std::string& error);
// Optional second stage: spatially subdivide only unconverged valid samples.
// additional_evaluations is a separate PER-SAMPLE budget; evaluations in each
// result include both stages. Already converged results are unchanged. On a
// device/API error returns false and leaves every input result unchanged.
// Adaptive scheduling is on the host; all quadrature evaluations run on CUDA.
bool RefineQelGpu(const QelParameters&, const std::vector<QelSample>&,
                  double relative_tolerance, unsigned additional_evaluations, int device,
                  std::vector<QelIntegralResult>&, std::string& error);
// Diagnostic entry point for independent, pointwise GENIE parity tests.
bool EvaluateQelGpu(const QelParameters&, const std::vector<QelSample>&,
                    const std::vector<double>& cosine, const std::vector<double>& phi,
                    int device, std::vector<double>& values, std::string& error);
}
#endif
