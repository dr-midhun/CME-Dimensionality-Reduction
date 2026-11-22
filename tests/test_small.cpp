// Validation against the thesis Michaelis-Menten example (eqs. 4.77-4.85).
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "cme/eigenbasis.hpp"
#include "cme/network.hpp"
#include "cme/state_space.hpp"
#include "cme/transition.hpp"

using namespace cme;
static int failures = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::printf("FAIL: %s\n", msg); ++failures; } else std::printf("ok:   %s\n", msg); } while (0)

int main(int argc, char** argv) {
    Backend backend = (argc > 1 && std::string(argv[1]) == "gpu") ? Backend::Cuda : Backend::Cpu;
    Network net;
    net.species = {"S1", "S2", "S3", "S4"};
    //            S1  S2  S3  S4
    net.R = {{1, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 1, 0}, {0, 1, 0, 1}};
    net.S = {{-1, -1, 1, 0}, {1, 1, -1, 0}, {0, 1, -1, 1}, {0, -1, 1, -1}};
    net.rate = {1000, 500, 5, 1};
    net.C = {{1, 0, 1, 1}, {0, 1, 1, 0}};
    net.x0 = {3, 3, 0, 0};

    StateSpace ss = generate_state_space(net);
    CHECK(ss.n() == 10, "state space has 10 states (thesis eq. 4.77)");

    Csr D = build_transition_matrix(net, ss);
    CHECK(D.nnz() == 34, "matrix has 34 stored entries (10 diagonal + 24 transitions)");
    // Columns of a CME generator sum to zero.
    std::vector<double> colsum(D.n, 0.0);
    double maxdiag = 0.0;
    for (int i = 0; i < D.n; ++i)
        for (int e = D.row_ptr[i]; e < D.row_ptr[i + 1]; ++e) {
            colsum[D.col[e]] += D.val[e];
            if (D.col[e] == i) maxdiag = std::max(maxdiag, std::fabs(D.val[e]));
        }
    double worst = 0;
    for (double c : colsum) worst = std::max(worst, std::fabs(c));
    CHECK(worst < 1e-9, "all column sums are zero");
    CHECK(std::fabs(maxdiag - 9000.0) < 1e-9, "largest outflow is 9000 (thesis eq. 4.79)");

    EigenOptions opt;
    opt.d = 4;
    opt.tol = 1e-12;
    EigenResult r = compute_eigenbasis(D, backend, opt);
    CHECK(r.converged, "block power iteration converged");
    CHECK(r.residual < 1e-9, "D B = B Dred holds (residual < 1e-9)");
    if (!r.eig_re.empty()) {
        std::vector<double> ev = r.eig_re;
        std::sort(ev.begin(), ev.end(), [](double a, double b) { return a > b; });
        std::printf("      eigenvalues: %.4f %.4f %.4f %.4f\n", ev[0], ev[1], ev[2], ev[3]);
        CHECK(std::fabs(ev[0]) < 1e-6, "lambda1 = 0 (thesis eq. 4.81)");
        CHECK(std::fabs(ev[1] + 7.2) < 0.1, "lambda2 = -7.2 (thesis eq. 4.82)");
        CHECK(std::fabs(ev[2] + 12.7) < 0.1, "lambda3 = -12.7 (thesis eq. 4.83)");
        CHECK(std::fabs(ev[3] + 18.1) < 0.1, "lambda4 = -18.1 (thesis eq. 4.84)");
    } else {
        std::puts("skip: eigenvalue checks need LAPACK (make LAPACK=1)");
    }
    // Fast modes (largest |lambda|): reference values from dense LAPACK/NumPy.
    {
        EigenOptions fo; fo.d = 2; fo.mode = Mode::Fast; fo.tol = 1e-12;
        EigenResult f = compute_eigenbasis(D, backend, fo);
        CHECK(f.converged, "fast-mode iteration converged");
        if (!f.eig_re.empty()) {
            std::vector<double> ev = f.eig_re;
            std::sort(ev.begin(), ev.end());
            CHECK(std::fabs(ev[0] + 9919.0097) < 1e-2 && std::fabs(ev[1] + 6769.2984) < 1e-2,
                  "fast eigenvalues -9919.0097, -6769.2984 (dense reference)");
        }
    }
    // Larger population (231 states): plain power iteration is hopeless (sigma/gap ~ 4e4),
    // the Chebyshev filter with a cutoff between lambda_4 = -51.5 and lambda_5 = -66.3 is not.
    {
        Network big = net;
        big.x0 = {20, 20, 0, 0};
        StateSpace ss2 = generate_state_space(big);
        CHECK(ss2.n() == 231, "20/20 population has 231 states");
        Csr D2 = build_transition_matrix(big, ss2);
        EigenOptions co; co.d = 4; co.cutoff = 60.0; co.degree = 100; co.tol = 1e-10;
        EigenResult c = compute_eigenbasis(D2, backend, co);
        CHECK(c.converged, "Chebyshev-filtered iteration converged");
        if (!c.eig_re.empty()) {
            std::vector<double> ev = c.eig_re;
            std::sort(ev.begin(), ev.end(), [](double a, double b) { return a > b; });
            std::printf("      eigenvalues: %.4f %.4f %.4f %.4f\n", ev[0], ev[1], ev[2], ev[3]);
            CHECK(std::fabs(ev[0]) < 1e-5 && std::fabs(ev[1] + 18.520962) < 1e-3 &&
                  std::fabs(ev[2] + 35.684233) < 1e-3 && std::fabs(ev[3] + 51.5383) < 1e-3,
                  "eigenvalues 0, -18.5210, -35.6842, -51.5383 (dense reference)");
        }
    }
    std::printf("%s (%d failures)\n", failures ? "FAILED" : "ALL PASSED", failures);
    return failures ? 1 : 0;
}
