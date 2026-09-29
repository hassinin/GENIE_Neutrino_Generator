#ifndef GENIE_QEL_GPU_UTILS_H
#define GENIE_QEL_GPU_UTILS_H
#include "gpu_spline/qel_integrand.h"
#include <string>
namespace genie {
class Interaction;
class XSecAlgorithmI;
namespace utils {
// These routines only marshal supported, configured GENIE models. They never
// sample a nucleon, change the seed, or substitute a different form-factor model.
bool PrepareQelGpuParameters(const XSecAlgorithmI*, const Interaction&,
                              gpu_spline::QelParameters&, std::string& reason);
bool PrepareQelGpuSample(const XSecAlgorithmI*, const Interaction&,
                         const gpu_spline::QelParameters&, gpu_spline::QelSample&);
}
}
#endif
