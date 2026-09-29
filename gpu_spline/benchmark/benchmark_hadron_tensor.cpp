#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include "gpu_spline/gpu_hadron_tensor.h"

int main(int argc, char** argv) {
    std::string tensor_file = "/home/karimhassinin/GENIE_gpu_support/data/evgen/hadron_tensors/nieves/HadTensor120-Nieves-1000060120-FullAll-20150210.dat";
    int num_queries = 1000000;

    if (argc > 1) tensor_file = argv[1];
    if (argc > 2) num_queries = std::atoi(argv[2]);

    std::cout << "================================================================" << std::endl;
    std::cout << "   GPU Hadron Tensor Grid Interpolation & Physics Benchmark     " << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << "Tensor table : " << tensor_file << std::endl;
    std::cout << "Queries      : " << num_queries << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;

    gpuspline::GpuHadronTensor tensor(0);
    if (!tensor.LoadFile(tensor_file)) {
        std::cerr << "Failed to load hadron tensor file!" << std::endl;
        return 1;
    }

    if (!tensor.UploadToDevice()) {
        std::cerr << "Failed to upload hadron tensor to GPU!" << std::endl;
        return 1;
    }

    // Set up realistic kinematic parameters: 2.5 GeV nu_mu CC on nucleus
    int probe_pdg = 14;      // nu_mu
    double E_probe = 2.5;    // 2.5 GeV
    double m_probe = 0.0;
    double ml = 0.10565837;  // muon mass in GeV
    double Q_value = 0.0168; // nuclear Q-value in GeV

    std::cout << "\n[1/3] Generating " << num_queries << " synthetic phase space points..." << std::endl;
    std::vector<gpuspline::MecQuery> queries(num_queries);
    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> dist_Tl(0.05, E_probe - ml - 0.05);
    std::uniform_real_distribution<double> dist_cos(-1.0, 1.0);

    for (int i = 0; i < num_queries; ++i) {
        queries[i].probe_pdg = probe_pdg;
        queries[i].E_probe = E_probe;
        queries[i].m_probe = m_probe;
        queries[i].Tl = dist_Tl(rng);
        queries[i].cos_l = dist_cos(rng);
        queries[i].ml = ml;
        queries[i].Q_value = Q_value;
    }

    std::cout << "[2/3] Running CPU reference evaluation..." << std::endl;
    std::vector<double> cpu_results(num_queries, 0.0);
    auto t_cpu_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_queries; ++i) {
        const auto& q = queries[i];
        cpu_results[i] = tensor.DiffXSecHost(
            q.probe_pdg, q.E_probe, q.m_probe,
            q.Tl, q.cos_l, q.ml, q.Q_value,
            false, 0.9742);
    }
    auto t_cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time_ms = std::chrono::duration<double, std::milli>(t_cpu_end - t_cpu_start).count();

    std::cout << "[3/3] Running GPU accelerated batch evaluation..." << std::endl;
    std::vector<double> gpu_results(num_queries, 0.0);
    // Warmup
    tensor.EvalDiffXSecBatch(queries.data(), gpu_results.data(), std::min(num_queries, 10000), false, 0.9742);

    auto t_gpu_start = std::chrono::high_resolution_clock::now();
    tensor.EvalDiffXSecBatch(queries.data(), gpu_results.data(), num_queries, false, 0.9742);
    auto t_gpu_end = std::chrono::high_resolution_clock::now();
    double gpu_time_ms = std::chrono::duration<double, std::milli>(t_gpu_end - t_gpu_start).count();

    // Device-resident pure kernel measurement
    gpuspline::MecQuery* d_queries = nullptr;
    double* d_out = nullptr;
    gpuMalloc((void**)&d_queries, num_queries * sizeof(gpuspline::MecQuery));
    gpuMalloc((void**)&d_out, num_queries * sizeof(double));
    gpuMemcpy(d_queries, queries.data(), num_queries * sizeof(gpuspline::MecQuery), gpuMemcpyHostToDevice);

    auto t_k_start = std::chrono::high_resolution_clock::now();
    tensor.EvalDiffXSecDevice(d_queries, d_out, num_queries, false, 0.9742);
    auto t_k_end = std::chrono::high_resolution_clock::now();
    double kernel_time_ms = std::chrono::duration<double, std::milli>(t_k_end - t_k_start).count();

    gpuFree(d_queries);
    gpuFree(d_out);

    // Verification
    std::cout << "\n----------------------------------------------------------------" << std::endl;
    std::cout << "                   Numerical Precision Check                    " << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;

    int non_zero_count = 0;
    double max_abs_diff = 0.0;
    double max_rel_diff = 0.0;
    double max_xsec_val = 0.0;

    for (int i = 0; i < num_queries; ++i) {
        double c = cpu_results[i];
        double g = gpu_results[i];
        double diff = std::abs(c - g);
        if (diff > max_abs_diff) max_abs_diff = diff;
        if (c > 0.0) {
            non_zero_count++;
            if (c > max_xsec_val) max_xsec_val = c;
            double rel = diff / c;
            if (rel > max_rel_diff) max_rel_diff = rel;
        }
    }

    std::cout << "Physical non-zero cross sections: " << non_zero_count << " / " << num_queries << std::endl;
    std::cout << "Peak differential cross section : " << std::scientific << max_xsec_val << " GeV^-3" << std::endl;
    std::cout << "Max Absolute Difference          : " << std::scientific << max_abs_diff << std::endl;
    std::cout << "Max Relative Difference          : " << std::scientific << max_rel_diff << std::endl;
    std::cout << "Status                           : " 
              << ((max_rel_diff < 1e-12) ? "PASSED (EXACT AGREEMENT)" : "FAILED") << std::endl;

    std::cout << "\n----------------------------------------------------------------" << std::endl;
    std::cout << "                   Throughput & Speedup Results                 " << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "CPU Reference Time   : " << std::setw(8) << cpu_time_ms << " ms  ("
              << (num_queries / cpu_time_ms / 1000.0) << " M evals/sec)  [1.0x baseline]" << std::endl;
    std::cout << "GPU (End-to-End PCIe): " << std::setw(8) << gpu_time_ms << " ms  ("
              << (num_queries / gpu_time_ms / 1000.0) << " M evals/sec)  [" 
              << (cpu_time_ms / gpu_time_ms) << "x speedup]" << std::endl;
    std::cout << "GPU (Pure Kernel)    : " << std::setw(8) << kernel_time_ms << " ms  ("
              << (num_queries / kernel_time_ms / 1000.0) << " M evals/sec)  [" 
              << (cpu_time_ms / kernel_time_ms) << "x speedup]" << std::endl;
    std::cout << "================================================================" << std::endl;

    // Test rejection sampling across 1,000 events
    int num_events_test = 1000;
    std::cout << "\n[Rejection Sampling Benchmark] Generating " << num_events_test << " MEC events on GPU..." << std::endl;
    int accepted_events = 0;
    auto t_rej_start = std::chrono::high_resolution_clock::now();
    for (int ev = 0; ev < num_events_test; ++ev) {
        gpuspline::MecKinematicsCandidate cand;
        bool ok = tensor.SampleKinematicsGPU(
            probe_pdg, E_probe, m_probe, ml,
            0.05, E_probe - ml, -1.0, 1.0,
            Q_value, max_xsec_val * 1.1, false, 0.9742,
            (unsigned long long)(1000 + ev), cand, 4096);
        if (ok && cand.accepted) accepted_events++;
    }
    auto t_rej_end = std::chrono::high_resolution_clock::now();
    double rej_time_ms = std::chrono::duration<double, std::milli>(t_rej_end - t_rej_start).count();

    std::cout << "Accepted events: " << accepted_events << " / " << num_events_test 
              << " (" << (accepted_events * 100.0 / num_events_test) << "%)" << std::endl;
    std::cout << "Total time     : " << rej_time_ms << " ms (" 
              << (rej_time_ms / num_events_test) << " ms/event, "
              << (num_events_test / (rej_time_ms / 1000.0)) << " events/sec)" << std::endl;
    std::cout << "================================================================" << std::endl;

    // Test 2D numerical quadrature integration across multiple neutrino energies
    std::cout << "\n[2D Quadrature Benchmark] Computing total cross-section on GPU..." << std::endl;
    double total_xsec_gpu = 0.0;
    auto t_int_start = std::chrono::high_resolution_clock::now();
    int n_evals = 100;
    double Q3Max = 1.2;
    for (int k = 0; k < n_evals; ++k) {
        double E_k = 0.2 + k * (4.8 / n_evals);
        double T_min = 0.0;
        double costh_min = -1.0;
        if (E_k >= Q3Max) {
            T_min = std::sqrt(ml * ml + (E_k - Q3Max) * (E_k - Q3Max)) - ml;
            costh_min = std::sqrt(1.0 - (Q3Max / E_k) * (Q3Max / E_k));
        }
        tensor.Integrate2DGPU(
            probe_pdg, E_k, m_probe, ml,
            0.0 /* Delta_Q_value */, T_min, E_k - ml, costh_min, 1.0,
            Q3Max, 0.0 /* Q2min */,
            true /* use_rosenbluth */, 0.9742,
            total_xsec_gpu, 257, 257);
    }
    auto t_int_end = std::chrono::high_resolution_clock::now();
    double int_time_ms = std::chrono::duration<double, std::milli>(t_int_end - t_int_start).count();

    std::cout << "Computed " << n_evals << " energy knots in " << int_time_ms << " ms ("
              << (int_time_ms / n_evals) << " ms / knot)" << std::endl;
    std::cout << "Last knot total xsec (E=" << (0.2 + (n_evals - 1) * (4.8 / n_evals)) << " GeV): " 
              << std::scientific << total_xsec_gpu << " GeV^-2 (" 
              << (total_xsec_gpu * 3.8937929e10) << " x 10^-38 cm^2)" << std::endl;
    std::cout << "Throughput: " << std::fixed << (n_evals / (int_time_ms / 1000.0)) << " knots/sec" << std::endl;
    std::cout << "================================================================" << std::endl;

    return 0;
}
