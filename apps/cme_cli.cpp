#include <omp.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

#include "cme/alg1_cme.hpp"
#include "cme/alg6_7_8_fsp.hpp"
#include "cme/alg9_10_11_12_preconditioned_fsp.hpp"
#include "cme/eigenbasis.hpp"
#include "cme/network.hpp"
#include "cme/reduction.hpp"
#include "cme/state_space.hpp"
#include "cme/transition.hpp"

using namespace cme;

static void usage() {
    std::puts(
        "usage: cme_cli <network.net> [options]\n"
        "  --d N                  basis dimension (default 2, max 64)\n"
        "  --mode slow|fast       slow: eigenvalues nearest 0 (default); fast: largest |lambda|\n"
        "  --backend cpu|gpu      backend to use for eigenbasis (default: cpu)\n"
        "  --tol X                stop when max|K B - B Kred| < X (default 1e-10)\n"
        "  --max-iter N           outer iterations (default 200000)\n"
        "  --inner N              plain power mode: products between normalisations (default 4)\n"
        "  --cutoff X             Chebyshev mode: wanted |lambda| < X (slow) or > X (fast); thesis eq. 4.55\n"
        "  --degree N             Chebyshev polynomial degree (default 100)\n"
        "  --shift X              sigma; default Gershgorin bound 2*max|D_ii|\n"
        "  --out PREFIX           output prefix (default 'out')\n"
        "  --dump-matrix          write PREFIX.D.mtx (Matrix Market)\n"
        "  --dump-states          write PREFIX.states.csv\n"
        "  --reduce               perform Algorithm 4 reduction (computes X0, Qdd, back-projection)\n"
        "  --transform none|left  reduction transform (default 'left': left eigenbasis, preserving probability)\n"
        "  --solve-time T         time at which to evaluate approximate distribution P(T) under Algorithm 4\n"
        "  --alg1-bfs             run Algorithm 1 (breadth-first iterative CME state generation)\n"
        "  --fsp MODE             run FSP: 'classic' (Alg 6), 'fixed' (Alg 7), or 'variable' (Alg 8)\n"
        "  --preconditioned-fsp   run Algorithm 9 (transient multi-time-step FSP with preconditioning)\n"
        "  --t0 T                 initial simulation time for FSP (default 0.0)\n"
        "  --tf T                 final simulation time for FSP (default 1.0)\n"
        "  --eps X                error tolerance for FSP / preconditioning (default 1e-4)\n"
        "  --t-eps X              error dynamics timescale t_eps for preconditioning (default 0.01)\n"
        "  --num-steps N          number of time steps for fixed FSP (default 10)\n"
        "  --verbose\n"
        "outputs: PREFIX.basis.bin (float64, row-major n x d), PREFIX.pivots.txt,\n"
        "         PREFIX.dreduced.csv, PREFIX.eigenvalues.csv, PREFIX.eigenvectors.bin (if all real)\n"
        "         with --reduce: PREFIX.q_reduced.csv, PREFIX.x0.csv, PREFIX.perm.txt, [PREFIX.p_approx.csv]\n"
        "         with --fsp: PREFIX.fsp_prob.csv\n"
        "         with --preconditioned-fsp: PREFIX.prefsp_prob.csv, PREFIX.prefsp_steps.csv");
}

int main(int argc, char** argv) {
    if (argc < 2 || !std::strcmp(argv[1], "--help")) { usage(); return argc < 2 ? 1 : 0; }
    std::string path = argv[1], prefix = "out";
    EigenOptions opt;
    Backend backend = Backend::Cpu;
    bool dump_mtx = false, dump_states = false;
    bool run_reduction = false;
    bool run_alg1 = false;
    std::string fsp_mode;
    bool run_prefsp = false;
    double t0 = 0.0;
    double tf = 1.0;
    double eps = 1e-4;
    double t_eps = 0.01;
    int num_fsp_steps = 10;
    ReductionTransform red_transform = ReductionTransform::LeftEigenbasis;
    double solve_time = -1.0;

    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(1); }
            return argv[++i];
        };
        if (a == "--d") opt.d = std::atoi(next());
        else if (a == "--mode") opt.mode = std::string(next()) == "fast" ? Mode::Fast : Mode::Slow;
        else if (a == "--backend") backend = std::string(next()) == "gpu" ? Backend::Cuda : Backend::Cpu;
        else if (a == "--tol") opt.tol = std::atof(next());
        else if (a == "--max-iter") opt.max_iter = std::atoi(next());
        else if (a == "--inner") opt.inner = std::atoi(next());
        else if (a == "--cutoff") opt.cutoff = std::atof(next());
        else if (a == "--degree") opt.degree = std::atoi(next());
        else if (a == "--shift") opt.shift = std::atof(next());
        else if (a == "--out") prefix = next();
        else if (a == "--dump-matrix") dump_mtx = true;
        else if (a == "--dump-states") dump_states = true;
        else if (a == "--reduce") run_reduction = true;
        else if (a == "--transform") {
            std::string tval = next();
            if (tval == "none") red_transform = ReductionTransform::None;
            else if (tval == "left") red_transform = ReductionTransform::LeftEigenbasis;
            else { std::fprintf(stderr, "unknown transform %s\n", tval.c_str()); return 1; }
        }
        else if (a == "--solve-time") {
            solve_time = std::atof(next());
            run_reduction = true;
        }
        else if (a == "--alg1-bfs") run_alg1 = true;
        else if (a == "--fsp") fsp_mode = next();
        else if (a == "--preconditioned-fsp") run_prefsp = true;
        else if (a == "--t0") t0 = std::atof(next());
        else if (a == "--tf") tf = std::atof(next());
        else if (a == "--eps") eps = std::atof(next());
        else if (a == "--t-eps") t_eps = std::atof(next());
        else if (a == "--num-steps") num_fsp_steps = std::atoi(next());
        else if (a == "--verbose") opt.verbose = true;
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); usage(); return 1; }
    }

    try {
        Network net = parse_network(path);

        // Optional Algorithm 1 run
        if (run_alg1) {
            std::puts("--- Algorithm 1 (Iterative BFS CME Generation) ---");
            double t_bfs = omp_get_wtime();
            Alg1Result a1 = alg1_generate_cme(net);
            t_bfs = omp_get_wtime() - t_bfs;
            std::printf("Alg 1: discovered %zu states, %lld transitions (%.3fs)\n",
                        a1.states.size(), (long long)a1.D.nnz(), t_bfs);
        }

        // Optional FSP execution (Alg 6, 7, or 8)
        if (!fsp_mode.empty()) {
            std::printf("--- Finite State Projection (%s) ---\n", fsp_mode.c_str());
            double t_fsp = omp_get_wtime();
            FspOptions fsp_opt;
            fsp_opt.tf = tf;
            fsp_opt.eps = eps;
            fsp_opt.n_intervals = num_fsp_steps;
            FspResult fsp_res;
            if (fsp_mode == "classic") {
                fsp_res = alg6_fsp(net, fsp_opt);
            } else if (fsp_mode == "fixed") {
                fsp_res = alg7_multistep_fixed_fsp(net, fsp_opt);
            } else if (fsp_mode == "variable") {
                fsp_res = alg8_multistep_variable_fsp(net, fsp_opt);
            } else {
                throw std::invalid_argument("unknown fsp mode: " + fsp_mode + " (use classic, fixed, or variable)");
            }
            t_fsp = omp_get_wtime() - t_fsp;
            std::printf("FSP converged: %s | states = %zu | error bound = %.2e | time = %.3fs\n",
                        fsp_res.converged ? "yes" : "NO", fsp_res.states.size(), fsp_res.error_bound, t_fsp);
            std::ofstream f(prefix + ".fsp_prob.csv");
            f.precision(17);
            for (double p : fsp_res.prob) f << p << "\n";
            std::printf("FSP distribution saved to %s.fsp_prob.csv\n", prefix.c_str());
        }

        // Optional Preconditioned FSP execution (Alg 9, 10, 11, 12)
        if (run_prefsp) {
            std::puts("--- Algorithm 9 (Transient Multi-Time-Step FSP with Preconditioning) ---");
            double t_prefsp = omp_get_wtime();
            PreconditionedFspResult prefsp_res = alg9_preconditioned_fsp(net, t0, tf, eps, t_eps, opt.d);
            t_prefsp = omp_get_wtime() - t_prefsp;
            std::printf("Preconditioned FSP converged: %s | subintervals = %d | union states = %zu | time = %.3fs\n",
                        prefsp_res.converged ? "yes" : "NO", prefsp_res.num_subintervals,
                        prefsp_res.full_states.size(), t_prefsp);

            {
                std::ofstream f(prefix + ".prefsp_prob.csv");
                f.precision(17);
                for (double p : prefsp_res.P_tf) f << p << "\n";
                std::printf("Preconditioned FSP continuous solution at tf=%.6g saved to %s.prefsp_prob.csv\n",
                            tf, prefix.c_str());
            }
            {
                std::ofstream f(prefix + ".prefsp_steps.csv");
                f << "step,t_start,t_end,tau,d,local_states\n";
                for (size_t s = 0; s < prefsp_res.steps.size(); ++s) {
                    const auto& step = prefsp_res.steps[s];
                    f << s << "," << step.t_start << "," << step.t_end << "," << step.tau
                      << "," << step.d << "," << step.states.size() << "\n";
                }
                std::printf("Step trajectory metadata saved to %s.prefsp_steps.csv\n", prefix.c_str());
            }
        }

        // Standard Algorithm 2 & 3 pipeline
        double t = omp_get_wtime();
        StateSpace ss = generate_state_space(net);
        double t_states = omp_get_wtime() - t;
        t = omp_get_wtime();
        Csr D = build_transition_matrix(net, ss);
        double t_mat = omp_get_wtime() - t;
        std::printf("states n = %lld   nnz = %lld   (state space %.3fs, matrix %.3fs)\n",
                    (long long)ss.n(), (long long)D.nnz(), t_states, t_mat);

        if (dump_mtx) {
            std::ofstream f(prefix + ".D.mtx");
            f << "%%MatrixMarket matrix coordinate real general\n" << D.n << " " << D.n << " " << D.nnz() << "\n";
            f.precision(17);
            for (int32_t i = 0; i < D.n; ++i)
                for (int32_t e = D.row_ptr[i]; e < D.row_ptr[i + 1]; ++e)
                    f << i + 1 << " " << D.col[e] + 1 << " " << D.val[e] << "\n";
        }
        if (dump_states) {
            std::ofstream f(prefix + ".states.csv");
            for (int s = 0; s < net.k(); ++s) f << net.species[s] << (s + 1 < net.k() ? "," : "\n");
            std::vector<int> N(net.k());
            for (int64_t v = 0; v < ss.n(); ++v) {
                ss.decode(ss.keys[v], N.data());
                for (int s = 0; s < net.k(); ++s) f << N[s] << (s + 1 < net.k() ? "," : "\n");
            }
        }

        EigenResult r = compute_eigenbasis(D, backend, opt);
        std::printf("backend %s, mode %s, d = %d, sigma = %.6g\n", backend == Backend::Cpu ? "cpu" : "gpu",
                    opt.mode == Mode::Slow ? "slow" : "fast", r.d, r.shift);
        std::printf("iterations %d  SpMM products %lld  converged %s  residual %.2e  iterate time %.3fs\n", r.iterations,
                    (long long)r.spmm_count, r.converged ? "yes" : "NO", r.residual, r.seconds_iterate);
        if (!r.warning.empty()) std::printf("WARNING: %s\n", r.warning.c_str());
        if (!r.eig_re.empty()) {
            std::puts("eigenvalues of the reduced matrix:");
            for (int j = 0; j < r.d; ++j) std::printf("  %.8g %+.8gi\n", r.eig_re[j], r.eig_im[j]);
        } else {
            std::puts("(built without LAPACK: eigenvalues not computed; see PREFIX.dreduced.csv)");
        }

        {
            std::ofstream f(prefix + ".basis.bin", std::ios::binary);
            f.write(reinterpret_cast<const char*>(r.basis.data()), r.basis.size() * sizeof(double));
        }
        {
            std::ofstream f(prefix + ".pivots.txt");
            for (auto p : r.pivots) f << p << "\n";
        }
        {
            std::ofstream f(prefix + ".dreduced.csv");
            f.precision(17);
            for (int i = 0; i < r.d; ++i)
                for (int j = 0; j < r.d; ++j) f << r.d_reduced[i * r.d + j] << (j + 1 < r.d ? "," : "\n");
        }
        if (!r.eig_re.empty()) {
            std::ofstream f(prefix + ".eigenvalues.csv");
            f.precision(17);
            for (int j = 0; j < r.d; ++j) f << r.eig_re[j] << "," << r.eig_im[j] << "\n";
            bool real = true;
            for (double x : r.eig_im) if (x != 0.0) real = false;
            if (real) {  // eigenvectors of D = B * (eigenvectors of the reduced matrix)
                std::vector<double> E(r.basis.size());
                const int d = r.d;
                #pragma omp parallel for
                for (int64_t i = 0; i < r.n; ++i)
                    for (int c = 0; c < d; ++c) {
                        double a = 0.0;
                        for (int m = 0; m < d; ++m) a += r.basis[i * d + m] * r.eigvec_reduced[c * d + m];
                        E[i * d + c] = a;
                    }
                std::ofstream g(prefix + ".eigenvectors.bin", std::ios::binary);
                g.write(reinterpret_cast<const char*>(E.data()), E.size() * sizeof(double));
            }
        }
        if (run_reduction) {
            std::puts("--- Algorithm 4 (Reduction Algorithm) ---");
            // Find initial state index in state space
            uint64_t key0 = ss.encode(net.x0.data());
            int64_t idx0 = ss.find(key0);
            std::vector<double> P0(ss.n(), 0.0);
            if (idx0 >= 0 && idx0 < ss.n()) {
                P0[idx0] = 1.0;
            } else {
                std::fill(P0.begin(), P0.end(), 1.0 / ss.n());
            }

            ReductionOptions red_opt;
            red_opt.d = opt.d;
            red_opt.transform = red_transform;
            red_opt.tol = opt.tol;
            red_opt.max_iter = opt.max_iter;
            red_opt.cutoff = opt.cutoff;
            red_opt.degree = opt.degree;
            red_opt.backend = backend;
            red_opt.verbose = opt.verbose;

            ReductionModel red_model = reduce_cme(D, P0, red_opt);
            std::printf("reduction complete: d = %d, transform = %s\n", red_model.d,
                        red_transform == ReductionTransform::LeftEigenbasis ? "left eigenbasis" : "none");

            {
                std::ofstream f(prefix + ".q_reduced.csv");
                f.precision(17);
                for (int i = 0; i < red_model.d; ++i)
                    for (int j = 0; j < red_model.d; ++j)
                        f << red_model.Q[i * red_model.d + j] << (j + 1 < red_model.d ? "," : "\n");
            }
            {
                std::ofstream f(prefix + ".x0.csv");
                for (int i = 0; i < red_model.d; ++i) f << red_model.X0[i] << "\n";
            }
            {
                std::ofstream f(prefix + ".perm.txt");
                for (int64_t p : red_model.perm) f << p << "\n";
            }
            if (solve_time >= 0.0) {
                std::vector<double> P_sol = red_model.solve(solve_time);
                std::ofstream f(prefix + ".p_approx.csv");
                f.precision(17);
                for (int64_t i = 0; i < ss.n(); ++i) f << P_sol[i] << "\n";
                std::printf("solved approximate distribution P(t=%.6g) written to %s.p_approx.csv\n",
                            solve_time, prefix.c_str());
            }
        }

        return r.converged ? 0 : 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
