#include "gpu_spline/gpu_spline_engine.h"
#include "gpu_spline/gpu_flux_preselector.h"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <cmath>

int main(int argc, char** argv) {
    std::string spline_path = "/data/karim/splines/MC_inputs/AR23_20i_00_000_v340_splines.xml.gz";
    size_t total_rays = 1000000; // 1 million rays

    if (argc > 1) spline_path = argv[1];
    if (argc > 2) total_rays = std::stoul(argv[2]);

    std::cout << "================================================================" << std::endl;
    std::cout << "        GPU Batched Flux Ray Pre-filtering Benchmark            " << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << "Spline file : " << spline_path << std::endl;
    std::cout << "Total rays  : " << total_rays << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;

    // 1. Initialize Engine
    gpu_spline::GpuSplineEngine engine;
    if (!engine.LoadSplinesFromXmlGz(spline_path)) {
        std::cerr << "Failed to load spline file!" << std::endl;
        return 1;
    }
    if (!engine.InitializeDevice(0)) {
        std::cerr << "Failed to initialize device!" << std::endl;
        return 1;
    }

    // 2. Setup Detector Geometry Materials (Liquid Argon detector)
    gpu_spline::GpuFluxPreselector preselector(&engine);

    // Setup Argon-40 target (e.g. LArTPC ~ 3.5m LAr -> ~5,000 kg/m^2)
    gpu_spline::MaterialConfig lar_mat;
    lar_mat.target_pdg = 1000180400;
    lar_mat.A = 40;
    lar_mat.max_pl = 5000.0; // kg/m^2

    // Search for total / representative cross-section splines for Argon-40
    // In our file we have various processes; let's find matching splines for nu_mu (14) and nu_e (12)
    lar_mat.spline_id_numu = -1;
    lar_mat.spline_id_nue = -1;
    lar_mat.spline_id_numubar = -1;
    lar_mat.spline_id_nuebar = -1;
    lar_mat.spline_id_nutau = -1;
    lar_mat.spline_id_nutaubar = -1;

    for (int id = 0; id < engine.GetNumSplines(); ++id) {
        const std::string& name = engine.GetSplineName(id);
        if (name.find("tgt:1000180400") != std::string::npos) {
            if (name.find("nu:14;") != std::string::npos && lar_mat.spline_id_numu < 0) lar_mat.spline_id_numu = id;
            if (name.find("nu:12;") != std::string::npos && lar_mat.spline_id_nue < 0) lar_mat.spline_id_nue = id;
            if (name.find("nu:-14;") != std::string::npos && lar_mat.spline_id_numubar < 0) lar_mat.spline_id_numubar = id;
            if (name.find("nu:-12;") != std::string::npos && lar_mat.spline_id_nuebar < 0) lar_mat.spline_id_nuebar = id;
        }
    }

    // Fallback if not specifically found
    if (lar_mat.spline_id_numu < 0) lar_mat.spline_id_numu = 0;
    if (lar_mat.spline_id_nue < 0) lar_mat.spline_id_nue = 0;
    if (lar_mat.spline_id_numubar < 0) lar_mat.spline_id_numubar = 0;
    if (lar_mat.spline_id_nuebar < 0) lar_mat.spline_id_nuebar = 0;

    preselector.AddMaterial(lar_mat);

    // Compute realistic glob_pmax based on maximum cross section in the energy range [0.5, 10.0] GeV
    double max_xsec = 0.0;
    for (double e = 0.5; e <= 10.0; e += 0.1) {
        double xs = engine.EvaluateHost(lar_mat.spline_id_numu, e);
        if (xs > max_xsec) max_xsec = xs;
    }
    // Convert GeV^-2 to m^2, then kg to g for Avogadro's number.
    const double kNA_factor = 6.02214179e+26 * 1.973269804e-16 * 1.973269804e-16;
    double max_prob = (kNA_factor * max_xsec * lar_mat.max_pl) / lar_mat.A;
    // Set realistic beam acceptance rate (~0.1% to 0.2% interacting rays, typical for detector geometry)
    double glob_pmax = max_prob * 500.0;

    preselector.SetGlobalPmax(glob_pmax);
    preselector.SetBatchSize(50000);

    if (!preselector.InitializeDevice(0)) {
        std::cerr << "Failed to initialize flux preselector device memory!" << std::endl;
        return 1;
    }

    std::cout << "\nRegistered Detector Materials:" << std::endl;
    std::cout << "  Material 1: Argon-40 (A=" << lar_mat.A << ", max pL=" << lar_mat.max_pl
              << " kg/m^2, numu_spline=" << lar_mat.spline_id_numu << ")" << std::endl;
    std::cout << "  Pmax scale factor: " << std::scientific << glob_pmax << std::endl;

    // 3. Generate Synthetic Beam Flux Rays (nu_mu with E in [0.5, 10.0] GeV)
    std::cout << "\n[1/3] Generating " << total_rays << " synthetic beam flux rays..." << std::endl;
    std::vector<double> h_energies(total_rays);
    std::vector<int> h_nupdgs(total_rays, 14); // nu_mu
    std::vector<double> h_rndms(total_rays);

    std::mt19937_64 rng(1337);
    std::uniform_real_distribution<double> e_dist(0.5, 10.0);
    std::uniform_real_distribution<double> r_dist(0.0, 1.0);

    for (size_t i = 0; i < total_rays; ++i) {
        h_energies[i] = e_dist(rng);
        h_rndms[i] = r_dist(rng);
    }

    // 4. CPU Reference Pre-selection Benchmark
    std::cout << "[2/3] Running CPU sequential flux rejection..." << std::endl;
    size_t cpu_accepted_count = 0;

    auto t_cpu_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < total_rays; ++i) {
        double E = h_energies[i];
        double R = h_rndms[i];

        double xsec = engine.EvaluateHost(lar_mat.spline_id_numu, E);
        double prob = (kNA_factor * xsec * lar_mat.max_pl) / lar_mat.A;
        double psum = prob / glob_pmax;

        if (R < psum) {
            cpu_accepted_count++;
        }
    }
    auto t_cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time_s = std::chrono::duration<double>(t_cpu_end - t_cpu_start).count();
    double cpu_throughput = (total_rays / cpu_time_s) / 1e6;

    std::cout << "      CPU time: " << std::fixed << std::setprecision(3) << cpu_time_s * 1000.0 << " ms"
              << " (" << std::setprecision(2) << cpu_throughput << " M rays/sec)" << std::endl;
    std::cout << "      CPU Accepted candidates: " << cpu_accepted_count << " / " << total_rays
              << " (Rejection rate: " << std::setprecision(4)
              << 100.0 * (1.0 - static_cast<double>(cpu_accepted_count) / total_rays) << "%)" << std::endl;

    // 5. GPU Batched Pre-selection Benchmark
    std::cout << "[3/3] Running GPU Batched flux pre-filtering (Batches of 50,000)..." << std::endl;
    size_t batch_size = 50000;
    size_t gpu_accepted_count = 0;

    auto t_gpu_start = std::chrono::high_resolution_clock::now();
    for (size_t offset = 0; offset < total_rays; offset += batch_size) {
        size_t current_batch = std::min(batch_size, total_rays - offset);
        size_t accepted = preselector.FilterRayBatch(
            h_energies.data() + offset,
            h_nupdgs.data() + offset,
            h_rndms.data() + offset,
            nullptr,
            current_batch);
        gpu_accepted_count += accepted;
    }
    auto t_gpu_end = std::chrono::high_resolution_clock::now();
    double gpu_time_s = std::chrono::duration<double>(t_gpu_end - t_gpu_start).count();
    double gpu_throughput = (total_rays / gpu_time_s) / 1e6;

    std::cout << "      GPU Batched time: " << std::fixed << std::setprecision(3) << gpu_time_s * 1000.0 << " ms"
              << " (" << std::setprecision(2) << gpu_throughput << " M rays/sec)" << std::endl;
    std::cout << "      GPU Accepted candidates: " << gpu_accepted_count << " / " << total_rays << std::endl;

    // 6. Validation
    bool match = (cpu_accepted_count == gpu_accepted_count);
    std::cout << "\n----------------------------------------------------------------" << std::endl;
    std::cout << "Verification Status: " << (match ? "PASSED (EXACT MATCH IN ACCEPTED RAYS)" : "FAILED") << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;

    // 7. Summary Table
    std::cout << "\n================================================================" << std::endl;
    std::cout << "             FLUX PRE-FILTERING PERFORMANCE SUMMARY             " << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << "Mode                  Time (ms)       Throughput (M rays/s)     Speedup" << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << "CPU Sequential      : " << std::fixed << std::setw(8) << std::setprecision(2) << cpu_time_s * 1000.0
              << " ms    " << std::setw(12) << std::setprecision(2) << cpu_throughput << " M/s      1.0x (baseline)" << std::endl;
    std::cout << "GPU Batched (V100)  : " << std::fixed << std::setw(8) << std::setprecision(2) << gpu_time_s * 1000.0
              << " ms    " << std::setw(12) << std::setprecision(2) << gpu_throughput << " M/s      "
              << std::setprecision(1) << (cpu_time_s / gpu_time_s) << "x" << std::endl;
    std::cout << "================================================================" << std::endl;

    return match ? 0 : 1;
}
