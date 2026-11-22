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
        "  --alg N                run Thesis Algorithm 1-12 individually and export its artifacts\n"
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
    int target_alg = 0;

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
        else if (a == "--alg") target_alg = std::atoi(next());
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

        // Dedicated execution for any individual Thesis Algorithm (1 through 12)
        if (target_alg > 0) {
            if (target_alg < 1 || target_alg > 12) {
                std::fprintf(stderr, "error: invalid algorithm %d (must be between 1 and 12)\n", target_alg);
                return 1;
            }
            std::printf("=======================================================\n");
            std::printf("Executing Thesis Algorithm %d on %s\n", target_alg, path.c_str());
            std::printf("=======================================================\n");

            if (target_alg == 1) {
                // Algorithm 1: Iterative BFS CME Generation
                double t0_w = omp_get_wtime();
                Alg1Result a1 = alg1_generate_cme(net);
                double el = omp_get_wtime() - t0_w;
                std::printf("Alg 1: Discovered %zu states, %lld transitions (%.3fs)\n",
                            a1.states.size(), (long long)a1.D.nnz(), el);

                std::string states_file = prefix + ".states.csv";
                std::ofstream fs(states_file);
                for (int s = 0; s < net.k(); ++s) fs << net.species[s] << (s + 1 < net.k() ? "," : "\n");
                for (const auto& st : a1.states) {
                    for (int s = 0; s < net.k(); ++s) fs << st[s] << (s + 1 < net.k() ? "," : "\n");
                }
                std::printf("  -> States written to %s\n", states_file.c_str());

                std::string mtx_file = prefix + ".D.mtx";
                std::ofstream fm(mtx_file);
                fm << "%%MatrixMarket matrix coordinate real general\n" << a1.D.n << " " << a1.D.n << " " << a1.D.nnz() << "\n";
                fm.precision(17);
                for (int32_t i = 0; i < a1.D.n; ++i)
                    for (int32_t e = a1.D.row_ptr[i]; e < a1.D.row_ptr[i + 1]; ++e)
                        fm << i + 1 << " " << a1.D.col[e] + 1 << " " << a1.D.val[e] << "\n";
                std::printf("  -> Generator matrix D written to %s\n", mtx_file.c_str());
                return 0;
            }
            if (target_alg == 2) {
                // Algorithm 2: State Space Generation
                double t0_w = omp_get_wtime();
                StateSpace ss = generate_state_space(net);
                double el = omp_get_wtime() - t0_w;
                std::printf("Alg 2: Generated %lld reachable states (%.3fs)\n", (long long)ss.n(), el);

                std::string states_file = prefix + ".states.csv";
                std::ofstream fs(states_file);
                for (int s = 0; s < net.k(); ++s) fs << net.species[s] << (s + 1 < net.k() ? "," : "\n");
                std::vector<int> N(net.k());
                for (int64_t v = 0; v < ss.n(); ++v) {
                    ss.decode(ss.keys[v], N.data());
                    for (int s = 0; s < net.k(); ++s) fs << N[s] << (s + 1 < net.k() ? "," : "\n");
                }
                std::printf("  -> States written to %s\n", states_file.c_str());
                return 0;
            }
            if (target_alg == 3) {
                // Algorithm 3: Transition Matrix Generation
                double t0_w = omp_get_wtime();
                StateSpace ss = generate_state_space(net);
                Csr D = build_transition_matrix(net, ss);
                double el = omp_get_wtime() - t0_w;
                std::printf("Alg 3: Built transition matrix D (%d x %d, %lld non-zeros, %.3fs)\n",
                            D.n, D.n, (long long)D.nnz(), el);

                std::string mtx_file = prefix + ".D.mtx";
                std::ofstream fm(mtx_file);
                fm << "%%MatrixMarket matrix coordinate real general\n" << D.n << " " << D.n << " " << D.nnz() << "\n";
                fm.precision(17);
                for (int32_t i = 0; i < D.n; ++i)
                    for (int32_t e = D.row_ptr[i]; e < D.row_ptr[i + 1]; ++e)
                        fm << i + 1 << " " << D.col[e] + 1 << " " << D.val[e] << "\n";
                std::printf("  -> Generator matrix D written to %s\n", mtx_file.c_str());
                return 0;
            }
            if (target_alg == 4) {
                // Algorithm 4: Dimensionality Reduction
                StateSpace ss = generate_state_space(net);
                Csr D = build_transition_matrix(net, ss);
                uint64_t key0 = ss.encode(net.x0.data());
                int64_t idx0 = ss.find(key0);
                std::vector<double> P0(ss.n(), 0.0);
                if (idx0 >= 0 && idx0 < ss.n()) P0[idx0] = 1.0;
                else P0[0] = 1.0;

                ReductionOptions red_opt;
                red_opt.d = opt.d > 0 ? opt.d : 4;
                red_opt.transform = red_transform;
                red_opt.tol = opt.tol;
                red_opt.max_iter = opt.max_iter;
                red_opt.cutoff = opt.cutoff;
                red_opt.degree = opt.degree;
                red_opt.backend = backend;
                red_opt.verbose = opt.verbose;

                double t0_w = omp_get_wtime();
                ReductionModel rm = reduce_cme(D, P0, red_opt);
                double el = omp_get_wtime() - t0_w;
                std::printf("Alg 4: Dimensionality reduction completed: n = %d -> d = %d (%.3fs)\n",
                            D.n, rm.d, el);

                std::string q_file = prefix + ".q_reduced.csv";
                std::ofstream fq(q_file);
                fq.precision(17);
                for (int i = 0; i < rm.d; ++i)
                    for (int j = 0; j < rm.d; ++j)
                        fq << rm.Q[i * rm.d + j] << (j + 1 < rm.d ? "," : "\n");
                std::printf("  -> Reduced generator Q_dd written to %s\n", q_file.c_str());

                std::string x0_file = prefix + ".x0.csv";
                std::ofstream fx(x0_file);
                fx.precision(17);
                for (int i = 0; i < rm.d; ++i) fx << rm.X0[i] << "\n";
                std::printf("  -> Projected initial condition X(0) written to %s\n", x0_file.c_str());

                std::string perm_file = prefix + ".perm.txt";
                std::ofstream fp(perm_file);
                for (int64_t p : rm.perm) fp << p << "\n";
                std::printf("  -> Pivot permutation written to %s\n", perm_file.c_str());

                double st = solve_time >= 0.0 ? solve_time : 0.1;
                std::vector<double> P_sol = rm.solve(st);
                std::string pa_file = prefix + ".p_approx.csv";
                std::ofstream fpa(pa_file);
                fpa.precision(17);
                for (int64_t i = 0; i < ss.n(); ++i) fpa << P_sol[i] << "\n";
                std::printf("  -> Approximate distribution P(t=%.4f) written to %s\n", st, pa_file.c_str());
                return 0;
            }
            if (target_alg == 5) {
                // Algorithm 5: Eigenbasis Calculation
                StateSpace ss = generate_state_space(net);
                Csr D = build_transition_matrix(net, ss);
                if (opt.d <= 0) opt.d = 4;
                double t0_w = omp_get_wtime();
                EigenResult r = compute_eigenbasis(D, backend, opt);
                double el = omp_get_wtime() - t0_w;
                std::printf("Alg 5: Eigenbasis computed: d = %d, iterations = %d, residual = %.2e (%.3fs)\n",
                            r.d, r.iterations, r.residual, el);

                std::string basis_file = prefix + ".basis.bin";
                std::ofstream fb(basis_file, std::ios::binary);
                fb.write(reinterpret_cast<const char*>(r.basis.data()), r.basis.size() * sizeof(double));
                std::printf("  -> Basis matrix B written to %s\n", basis_file.c_str());

                std::string pivots_file = prefix + ".pivots.txt";
                std::ofstream fp(pivots_file);
                for (auto p : r.pivots) fp << p << "\n";
                std::printf("  -> Pivot states written to %s\n", pivots_file.c_str());

                std::string dred_file = prefix + ".dreduced.csv";
                std::ofstream fd(dred_file);
                fd.precision(17);
                for (int i = 0; i < r.d; ++i)
                    for (int j = 0; j < r.d; ++j)
                        fd << r.d_reduced[i * r.d + j] << (j + 1 < r.d ? "," : "\n");
                std::printf("  -> Reduced matrix D_red written to %s\n", dred_file.c_str());

                if (!r.eig_re.empty()) {
                    std::string eig_file = prefix + ".eigenvalues.csv";
                    std::ofstream fe(eig_file);
                    fe.precision(17);
                    for (int j = 0; j < r.d; ++j) fe << r.eig_re[j] << "," << r.eig_im[j] << "\n";
                    std::printf("  -> Eigenvalues written to %s\n", eig_file.c_str());
                }
                return r.converged ? 0 : 2;
            }
            if (target_alg == 6) {
                // Algorithm 6: Classic FSP
                FspOptions fsp_opt;
                fsp_opt.tf = tf > 0 ? tf : 0.1;
                fsp_opt.eps = eps;
                double t0_w = omp_get_wtime();
                FspResult fsp_res = alg6_fsp(net, fsp_opt);
                double el = omp_get_wtime() - t0_w;
                std::printf("Alg 6 (Classic FSP): states = %zu, error bound = %.2e, converged = %s (%.3fs)\n",
                            fsp_res.states.size(), fsp_res.error_bound, fsp_res.converged ? "yes" : "NO", el);

                std::string prob_file = prefix + ".fsp_prob.csv";
                std::ofstream f(prob_file);
                f.precision(17);
                for (double p : fsp_res.prob) f << p << "\n";
                std::printf("  -> FSP probability distribution written to %s\n", prob_file.c_str());
                return fsp_res.converged ? 0 : 2;
            }
            if (target_alg == 7) {
                // Algorithm 7: Fixed Multi-Step FSP
                FspOptions fsp_opt;
                fsp_opt.tf = tf > 0 ? tf : 0.1;
                fsp_opt.eps = eps;
                fsp_opt.n_intervals = num_fsp_steps > 0 ? num_fsp_steps : 5;
                double t0_w = omp_get_wtime();
                FspResult fsp_res = alg7_multistep_fixed_fsp(net, fsp_opt);
                double el = omp_get_wtime() - t0_w;
                std::printf("Alg 7 (Fixed Multi-Step FSP): intervals = %d, states = %zu, error bound = %.2e (%.3fs)\n",
                            fsp_opt.n_intervals, fsp_res.states.size(), fsp_res.error_bound, el);

                std::string prob_file = prefix + ".fsp_prob.csv";
                std::ofstream f(prob_file);
                f.precision(17);
                for (double p : fsp_res.prob) f << p << "\n";
                std::printf("  -> Fixed FSP probability distribution written to %s\n", prob_file.c_str());
                return fsp_res.converged ? 0 : 2;
            }
            if (target_alg == 8) {
                // Algorithm 8: Variable Multi-Step FSP
                FspOptions fsp_opt;
                fsp_opt.tf = tf > 0 ? tf : 0.1;
                fsp_opt.eps = eps;
                double t0_w = omp_get_wtime();
                FspResult fsp_res = alg8_multistep_variable_fsp(net, fsp_opt);
                double el = omp_get_wtime() - t0_w;
                std::printf("Alg 8 (Variable Multi-Step FSP): states = %zu, error bound = %.2e, converged = %s (%.3fs)\n",
                            fsp_res.states.size(), fsp_res.error_bound, fsp_res.converged ? "yes" : "NO", el);

                std::string prob_file = prefix + ".fsp_prob.csv";
                std::ofstream f(prob_file);
                f.precision(17);
                for (double p : fsp_res.prob) f << p << "\n";
                std::printf("  -> Variable FSP probability distribution written to %s\n", prob_file.c_str());
                return fsp_res.converged ? 0 : 2;
            }
            if (target_alg == 9) {
                // Algorithm 9: Transient Preconditioned FSP
                int d_val = opt.d > 0 ? opt.d : 4;
                double t0_w = omp_get_wtime();
                PreconditionedFspResult prefsp = alg9_preconditioned_fsp(net, t0, tf, eps, t_eps, d_val);
                double el = omp_get_wtime() - t0_w;
                std::printf("Alg 9 (Preconditioned FSP): subintervals = %d, union states = %zu, converged = %s (%.3fs)\n",
                            prefsp.num_subintervals, prefsp.full_states.size(), prefsp.converged ? "yes" : "NO", el);

                std::string prob_file = prefix + ".prefsp_prob.csv";
                std::ofstream fp(prob_file);
                fp.precision(17);
                for (double p : prefsp.P_tf) fp << p << "\n";
                std::printf("  -> Final distribution P(tf=%.4f) written to %s\n", tf, prob_file.c_str());

                std::string steps_file = prefix + ".prefsp_steps.csv";
                std::ofstream fs(steps_file);
                fs << "step,t_start,t_end,tau,d,local_states\n";
                for (size_t s = 0; s < prefsp.steps.size(); ++s) {
                    const auto& step = prefsp.steps[s];
                    fs << s << "," << step.t_start << "," << step.t_end << "," << step.tau
                       << "," << step.d << "," << step.states.size() << "\n";
                }
                std::printf("  -> Step trajectory history written to %s\n", steps_file.c_str());
                return prefsp.converged ? 0 : 2;
            }
            if (target_alg == 10) {
                // Algorithm 10: Preconditioning Algorithm (optimal tau, reduced Q, Z0)
                StateSpace ss = generate_state_space(net);
                Csr D = build_transition_matrix(net, ss);
                uint64_t key0 = ss.encode(net.x0.data());
                int64_t idx0 = ss.find(key0);
                std::vector<double> P0(ss.n(), 0.0);
                if (idx0 >= 0 && idx0 < ss.n()) P0[idx0] = 1.0;
                else P0[0] = 1.0;

                int d_val = opt.d > 0 ? opt.d : 4;
                double t0_w = omp_get_wtime();
                PreconditionResult pr = alg10_precondition(D, P0, d_val, eps, t_eps);
                double el = omp_get_wtime() - t0_w;
                std::printf("Alg 10 (Preconditioning): optimal tau = %.6e, d = %d, error bound = %.2e (%.3fs)\n",
                            pr.tau, pr.d, eps, el);

                std::string tau_file = prefix + ".tau.txt";
                std::ofstream ft(tau_file);
                ft << "tau=" << pr.tau << "\n" << "d=" << pr.d << "\n" << "eps=" << eps << "\n" << "t_eps=" << t_eps << "\n";
                std::printf("  -> Optimal time step parameters written to %s\n", tau_file.c_str());

                std::string q_file = prefix + ".q_reduced.csv";
                std::ofstream fq(q_file);
                fq.precision(17);
                for (int i = 0; i < pr.d; ++i)
                    for (int j = 0; j < pr.d; ++j)
                        fq << pr.Q[i * pr.d + j] << (j + 1 < pr.d ? "," : "\n");
                std::printf("  -> Reduced generator Q written to %s\n", q_file.c_str());

                std::string z0_file = prefix + ".z0.csv";
                std::ofstream fz(z0_file);
                fz.precision(17);
                for (double z : pr.Z0) fz << z << "\n";
                std::printf("  -> Reduced initial condition Z0 written to %s\n", z0_file.c_str());

                std::string perm_file = prefix + ".perm.txt";
                std::ofstream fp(perm_file);
                for (int64_t p : pr.perm) fp << p << "\n";
                std::printf("  -> Re-indexing permutation written to %s\n", perm_file.c_str());
                return 0;
            }
            if (target_alg == 11) {
                // Algorithm 11: Subinterval Approximation
                StateSpace ss = generate_state_space(net);
                Csr D = build_transition_matrix(net, ss);
                uint64_t key0 = ss.encode(net.x0.data());
                int64_t idx0 = ss.find(key0);
                std::vector<double> P0(ss.n(), 0.0);
                if (idx0 >= 0 && idx0 < ss.n()) P0[idx0] = 1.0;
                else P0[0] = 1.0;

                int d_val = opt.d > 0 ? opt.d : 4;
                double t0_w = omp_get_wtime();
                PreconditionResult pr = alg10_precondition(D, P0, d_val, eps, t_eps);
                ApproximationResult ar = alg11_approximation(D, pr, pr.Z_tau);
                double el = omp_get_wtime() - t0_w;
                std::printf("Alg 11 (Subinterval Approximation): tau = %.6e, state count = %zu (%.3fs)\n",
                            pr.tau, ar.P_next.size(), el);

                std::string pnext_file = prefix + ".p_next.csv";
                std::ofstream fp(pnext_file);
                fp.precision(17);
                for (double p : ar.P_next) fp << p << "\n";
                std::printf("  -> Reconstructed probability distribution P_next written to %s\n", pnext_file.c_str());
                return 0;
            }
            if (target_alg == 12) {
                // Algorithm 12: Continuous Transient Solution
                int d_val = opt.d > 0 ? opt.d : 4;
                double t0_w = omp_get_wtime();
                PreconditionedFspResult prefsp = alg9_preconditioned_fsp(net, t0, tf, eps, t_eps, d_val);
                double el = omp_get_wtime() - t0_w;
                std::printf("Alg 12 (Continuous Transient Evaluation): evaluating continuous trajectory across [t0=%.4f, tf=%.4f] (%.3fs)\n",
                            t0, tf, el);

                std::string traj_file = prefix + ".trajectory.csv";
                std::ofstream ft(traj_file);
                ft << "time";
                for (size_t s = 0; s < prefsp.full_states.size(); ++s) ft << ",P_s" << s;
                ft << ",sum\n";
                ft.precision(17);

                int num_pts = 10;
                for (int step = 0; step <= num_pts; ++step) {
                    double q_t = t0 + step * (tf - t0) / num_pts;
                    std::vector<double> P_t = alg12_transient_solution(prefsp.steps, prefsp.full_states, q_t);
                    ft << q_t;
                    double sum = 0.0;
                    for (double p : P_t) {
                        ft << "," << p;
                        sum += p;
                    }
                    ft << "," << sum << "\n";
                }
                std::printf("  -> Continuous trajectory grid evaluated and written to %s\n", traj_file.c_str());
                return prefsp.converged ? 0 : 2;
            }
        }

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
