#include <omp.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

#include "cme/eigenbasis.hpp"
#include "cme/network.hpp"
#include "cme/state_space.hpp"
#include "cme/transition.hpp"

using namespace cme;

static void usage() {
    std::puts(
        "usage: cme_cli <network.net> [options]\n"
        "  --d N            basis dimension (default 2, max 64)\n"
        "  --mode slow|fast slow: eigenvalues nearest 0 (default); fast: largest |lambda|\n"
        "  --backend cpu|gpu\n"
        "  --tol X          stop when max|K B - B Kred| < X (default 1e-10)\n"
        "  --max-iter N     outer iterations (default 200000)\n"
        "  --inner N        plain power mode: products between normalisations (default 4)\n"
        "  --cutoff X       Chebyshev mode: wanted |lambda| < X (slow) or > X (fast); thesis eq. 4.55\n"
        "  --degree N       Chebyshev polynomial degree (default 100)\n"
        "  --shift X        sigma; default Gershgorin bound 2*max|D_ii|\n"
        "  --out PREFIX     output prefix (default 'out')\n"
        "  --dump-matrix    write PREFIX.D.mtx (Matrix Market)\n"
        "  --dump-states    write PREFIX.states.csv\n"
        "  --verbose\n"
        "outputs: PREFIX.basis.bin (float64, row-major n x d), PREFIX.pivots.txt,\n"
        "         PREFIX.dreduced.csv, PREFIX.eigenvalues.csv, PREFIX.eigenvectors.bin (if all real)");
}

int main(int argc, char** argv) {
    if (argc < 2 || !std::strcmp(argv[1], "--help")) { usage(); return argc < 2 ? 1 : 0; }
    std::string path = argv[1], prefix = "out";
    EigenOptions opt;
    Backend backend = Backend::Cpu;
    bool dump_mtx = false, dump_states = false;
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
        else if (a == "--verbose") opt.verbose = true;
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); usage(); return 1; }
    }
    try {
        double t = omp_get_wtime();
        Network net = parse_network(path);
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
        return r.converged ? 0 : 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
