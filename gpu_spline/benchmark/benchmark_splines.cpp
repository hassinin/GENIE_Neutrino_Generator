#include "gpu_spline/gpu_spline_engine.h"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <cmath>

int main(int argc, char** argv) {
    std::string spline_path = "/data/karim/splines/MC_inputs/AR23_20i_00_000_v340_splines.xml.gz";
    size_t n_queries = 2000000; // 2 million queries default

    if (argc > 1) {
        spline_path = argv[1];
    }
    if (argc > 2) {
        n_queries = std::stoul(argv[2]);
    }

    std::cout << "================================================================" << std::endl;
    std::cout << "       Portable C++/HIP GPU Cross-Section Spline Benchmark      " << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << "Spline file : " << spline_path << std::endl;
    std::cout << "Queries     : " << n_queries << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;

    gpu_spline::GpuSplineEngine engine;

    // 1. Load Splines
    if (!engine.LoadSplinesFromXmlGz(spline_path)) {
        std::cerr << "Failed to load spline file!" << std::endl;
        return 1;
    }

    // 2. Initialize GPU
    if (!engine.InitializeDevice(0)) {
        std::cerr << "Failed to initialize GPU!" << std::endl;
        return 1;
    }

    int n_splines = engine.GetNumSplines();
    std::cout << "Total splines loaded: " << n_splines << std::endl;
    std::cout << "Total knot intervals: " << engine.GetTotalIntervals() << std::endl;

    // 3. Generate Random Queries (E in [0.01, 100] GeV, random spline)
    std::cout << "\n[1/4] Generating " << n_queries << " synthetic evaluation queries..." << std::endl;
    std::vector<double> h_energies(n_queries);
    std::vector<int> h_spline_ids(n_queries);
    std::vector<double> h_results_cpu(n_queries, 0.0);
    std::vector<double> h_results_gpu(n_queries, 0.0);

    std::mt19937_64 rng(42);
    // Log-uniform distribution from 0.01 to 100 GeV
    std::uniform_real_distribution<double> log_e_dist(std::log10(0.01), std::log10(100.0));
    std::uniform_int_distribution<int> sp_dist(0, n_splines - 1);

    for (size_t i = 0; i < n_queries; ++i) {
        h_energies[i] = std::pow(10.0, log_e_dist(rng));
        h_spline_ids[i] = sp_dist(rng);
    }

    // 4. CPU Evaluation Benchmark
    std::cout << "[2/4] Running CPU reference evaluation..." << std::endl;
    auto t_cpu_start = std::chrono::high_resolution_clock::now();
    engine.EvaluateBatchHost(h_energies.data(), h_spline_ids.data(), h_results_cpu.data(), n_queries);
    auto t_cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time_s = std::chrono::duration<double>(t_cpu_end - t_cpu_start).count();
    double cpu_throughput = (n_queries / cpu_time_s) / 1e6; // Millions/sec

    std::cout << "      CPU time: " << std::fixed << std::setprecision(3) << cpu_time_s * 1000.0 << " ms"
              << " (" << std::setprecision(2) << cpu_throughput << " M evals/sec)" << std::endl;

    // 5. GPU Evaluation Benchmark (End-to-End with PCIe transfers)
    std::cout << "[3/4] Running GPU evaluation (End-to-End)..." << std::endl;
    // Warmup
    engine.EvaluateBatchGpu(h_energies.data(), h_spline_ids.data(), h_results_gpu.data(), std::min<size_t>(10000, n_queries));

    auto t_gpu_e2e_start = std::chrono::high_resolution_clock::now();
    engine.EvaluateBatchGpu(h_energies.data(), h_spline_ids.data(), h_results_gpu.data(), n_queries);
    auto t_gpu_e2e_end = std::chrono::high_resolution_clock::now();
    double gpu_e2e_time_s = std::chrono::duration<double>(t_gpu_e2e_end - t_gpu_e2e_start).count();
    double gpu_e2e_throughput = (n_queries / gpu_e2e_time_s) / 1e6;

    std::cout << "      GPU End-to-End time: " << std::fixed << std::setprecision(3) << gpu_e2e_time_s * 1000.0 << " ms"
              << " (" << std::setprecision(2) << gpu_e2e_throughput << " M evals/sec)" << std::endl;

    // 6. GPU Pure Kernel Benchmark (Device-only, simulating GPU-native workflows)
    std::cout << "[4/4] Running GPU pure kernel evaluation (Zero-Copy / Device-resident)..." << std::endl;
    double* d_energies = nullptr;
    int* d_spline_ids = nullptr;
    double* d_results = nullptr;
    size_t e_bytes = n_queries * sizeof(double);
    size_t id_bytes = n_queries * sizeof(int);

    GPU_CHECK(gpuMalloc(&d_energies, e_bytes));
    GPU_CHECK(gpuMalloc(&d_spline_ids, id_bytes));
    GPU_CHECK(gpuMalloc(&d_results, e_bytes));

    GPU_CHECK(gpuMemcpy(d_energies, h_energies.data(), e_bytes, gpuMemcpyHostToDevice));
    GPU_CHECK(gpuMemcpy(d_spline_ids, h_spline_ids.data(), id_bytes, gpuMemcpyHostToDevice));

    // Warmup
    engine.EvaluateDeviceData(d_energies, d_spline_ids, d_results, n_queries);
    GPU_CHECK(gpuDeviceSynchronize());

    int n_runs = 10;
    auto t_gpu_kernel_start = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < n_runs; ++r) {
        engine.EvaluateDeviceData(d_energies, d_spline_ids, d_results, n_queries);
    }
    GPU_CHECK(gpuDeviceSynchronize());
    auto t_gpu_kernel_end = std::chrono::high_resolution_clock::now();
    double gpu_kernel_time_s = std::chrono::duration<double>(t_gpu_kernel_end - t_gpu_kernel_start).count() / n_runs;
    double gpu_kernel_throughput = (n_queries / gpu_kernel_time_s) / 1e6;

    std::cout << "      GPU Kernel-only time: " << std::fixed << std::setprecision(3) << gpu_kernel_time_s * 1000.0 << " ms"
              << " (" << std::setprecision(2) << gpu_kernel_throughput << " M evals/sec)" << std::endl;

    GPU_CHECK(gpuFree(d_energies));
    GPU_CHECK(gpuFree(d_spline_ids));
    GPU_CHECK(gpuFree(d_results));

    // 7. Numerical Verification
    std::cout << "\n----------------------------------------------------------------" << std::endl;
    std::cout << "                   Numerical Precision Check                    " << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;
    double max_abs_diff = 0.0;
    double max_rel_diff = 0.0;
    size_t non_zero_count = 0;

    for (size_t i = 0; i < n_queries; ++i) {
        double cpu_val = h_results_cpu[i];
        double gpu_val = h_results_gpu[i];
        double diff = std::abs(gpu_val - cpu_val);

        if (diff > max_abs_diff) max_abs_diff = diff;

        if (cpu_val > 1e-30) {
            double rel = diff / cpu_val;
            if (rel > max_rel_diff) max_rel_diff = rel;
            non_zero_count++;
        }
    }

    std::cout << "Non-zero evaluations tested : " << non_zero_count << " / " << n_queries << std::endl;
    std::cout << "Max Absolute Difference     : " << std::scientific << std::setprecision(6) << max_abs_diff << std::endl;
    std::cout << "Max Relative Difference     : " << std::scientific << std::setprecision(6) << max_rel_diff << std::endl;

    bool passed = (max_rel_diff < 1e-9) || (max_abs_diff < 1e-15);
    std::cout << "Status                      : " << (passed ? "PASSED (EXACT AGREEMENT)" : "FAILED") << std::endl;

    // 8. Performance Summary Table
    std::cout << "\n================================================================" << std::endl;
    std::cout << "                     PERFORMANCE SUMMARY                        " << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << "Mode                  Time (ms)       Throughput (M evals/s)    Speedup" << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << "CPU Multi-threaded  : " << std::fixed << std::setw(8) << std::setprecision(2) << cpu_time_s * 1000.0
              << " ms    " << std::setw(12) << std::setprecision(2) << cpu_throughput << " M/s      1.0x (baseline)" << std::endl;
    std::cout << "GPU (End-to-End)    : " << std::fixed << std::setw(8) << std::setprecision(2) << gpu_e2e_time_s * 1000.0
              << " ms    " << std::setw(12) << std::setprecision(2) << gpu_e2e_throughput << " M/s      "
              << std::setprecision(1) << (cpu_time_s / gpu_e2e_time_s) << "x" << std::endl;
    std::cout << "GPU (Pure Kernel)   : " << std::fixed << std::setw(8) << std::setprecision(2) << gpu_kernel_time_s * 1000.0
              << " ms    " << std::setw(12) << std::setprecision(2) << gpu_kernel_throughput << " M/s      "
              << std::setprecision(1) << (cpu_time_s / gpu_kernel_time_s) << "x" << std::endl;
    std::cout << "================================================================" << std::endl;

    return passed ? 0 : 1;
}
