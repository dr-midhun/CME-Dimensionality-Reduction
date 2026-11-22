// Validation against the thesis Michaelis-Menten example (eqs. 4.77-4.85).
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "cme/alg1_cme.hpp"
#include "cme/alg2_state_space.hpp"
#include "cme/alg3_transition.hpp"
#include "cme/alg4_reduction.hpp"
#include "cme/alg5_eigenbasis.hpp"
#include "cme/alg6_7_8_fsp.hpp"
#include "cme/alg9_10_11_12_preconditioned_fsp.hpp"
#include "cme/network.hpp"

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
    // Algorithm 4 validation: MM model reduction (Thesis Section 4.4.2)
    {
        std::vector<double> P0(10, 0.1); // Thesis eq. 4.78: P(N, t=0) = [0.1, ..., 0.1]^T

        // 1. Left Eigenbasis Transformation (Thesis Section 4.2.1, eq. 4.94 - 4.100)
        ReductionOptions ropt_T;
        ropt_T.d = 4;
        ropt_T.transform = ReductionTransform::LeftEigenbasis;
        ropt_T.tol = 1e-12;
        ropt_T.backend = backend;
        ReductionModel red_T = reduce_cme(D, P0, ropt_T);

        // Check probability conservation in reduced initial condition X(0)
        double sumX0_T = 0.0;
        for (double v : red_T.X0) sumX0_T += v;
        CHECK(std::fabs(sumX0_T - 1.0) < 1e-6, "Alg 4 (LeftEigenbasis): sum(X0) == 1 (probability conservation)");

        // Check column sums of reduced generator Qdd are zero (Thesis eq. 4.95 note)
        for (int c = 0; c < red_T.d; ++c) {
            double csum = 0.0;
            for (int r = 0; r < red_T.d; ++r) csum += red_T.Q[r * red_T.d + c];
            CHECK(std::fabs(csum) < 1e-6, "Alg 4 (LeftEigenbasis): colsum(Q) == 0");
        }

        // Solve reduced system at t = 0.05 (well past t_eps ~ 0.0065)
        std::vector<double> P_approx_T = red_T.solve(0.05);
        double sum_P_T = 0.0;
        for (double v : P_approx_T) sum_P_T += v;
        CHECK(std::fabs(sum_P_T - 1.0) < 1e-6, "Alg 4 (LeftEigenbasis): sum(P_approx(t)) == 1");

        // 2. No Transformation (T = I) (Thesis Section 4.1, eq. 4.89 - 4.92)
        ReductionOptions ropt_noT;
        ropt_noT.d = 4;
        ropt_noT.transform = ReductionTransform::None;
        ropt_noT.tol = 1e-12;
        ropt_noT.backend = backend;
        ReductionModel red_noT = reduce_cme(D, P0, ropt_noT);

        std::vector<double> P_approx_noT = red_noT.solve(0.05);
        double sum_P_noT = 0.0;
        for (double v : P_approx_noT) sum_P_noT += v;
        CHECK(std::fabs(sum_P_noT - 1.0) < 1e-6, "Alg 4 (NoTransform): sum(P_approx(t)) == 1 (Corollary 1)");

        // Solutions under both transformations should agree with high accuracy for t >= t_eps
        double max_diff = 0.0;
        for (size_t i = 0; i < P_approx_T.size(); ++i) {
            max_diff = std::max(max_diff, std::fabs(P_approx_T[i] - P_approx_noT[i]));
        }
        CHECK(max_diff < 1e-5, "Alg 4: LeftEigenbasis and NoTransform approximations agree (max diff < 1e-5)");
    }

    // Algorithm 1 validation: Iterative breadth-first state space and CME generation (Thesis Section 3)
    {
        Alg1Result a1 = alg1_generate_cme(net);
        CHECK(a1.states.size() == 10, "Alg 1: generates 10 states for MM network");
        CHECK(a1.D.nnz() == 34, "Alg 1: generates 34 transitions in D");

        // Verify column sums of D from Alg 1 are all 0
        std::vector<double> csum(a1.D.n, 0.0);
        for (int i = 0; i < a1.D.n; ++i) {
            for (int e = a1.D.row_ptr[i]; e < a1.D.row_ptr[i + 1]; ++e) {
                csum[a1.D.col[e]] += a1.D.val[e];
            }
        }
        double worst = 0.0;
        for (double c : csum) worst = std::max(worst, std::fabs(c));
        CHECK(worst < 1e-9, "Alg 1: all column sums in D are zero");
    }

    // Algorithms 6, 7, 8 validation: Finite State Projection methods (Thesis Chapter 5)
    {
        FspOptions fopt;
        fopt.tf = 0.05;
        fopt.eps = 1e-2;

        // Alg 6: Classic FSP
        FspResult r6 = alg6_fsp(net, fopt);
        CHECK(r6.converged, "Alg 6 (Classic FSP): converged within error tolerance");
        CHECK(r6.error_bound <= fopt.eps, "Alg 6: error bound <= eps");

        // Alg 7: Fixed multi-step FSP
        fopt.n_intervals = 5;
        FspResult r7 = alg7_multistep_fixed_fsp(net, fopt);
        CHECK(r7.converged, "Alg 7 (Fixed Multi-step FSP): converged");
        CHECK(r7.error_bound <= fopt.eps * 1.05, "Alg 7: error bound <= eps");

        // Alg 8: Variable multi-step FSP
        FspResult r8 = alg8_multistep_variable_fsp(net, fopt);
        CHECK(r8.converged, "Alg 8 (Variable Multi-step FSP): converged");
        CHECK(r8.error_bound <= fopt.eps * 1.05, "Alg 8: error bound <= eps");
    }

    // Algorithms 9, 10, 11, 12 validation: Preconditioned Multi-Step FSP Pipeline (Thesis Chapter 5)
    {
        // Alg 10 & 11 direct check: Precondition and Approximation
        std::vector<double> P0(ss.n(), 0.0);
        P0[0] = 1.0;
        PreconditionResult precon = alg10_precondition(D, P0, 4, 1e-3, 0.01);
        CHECK(precon.tau > 0.0, "Alg 10 (Precondition): identified positive time step tau");
        CHECK(precon.d == 4, "Alg 10: reduced dimension d = 4");

        ApproximationResult approx = alg11_approximation(D, precon, precon.Z_tau);
        double sum_p = 0.0;
        for (double v : approx.P_next) sum_p += v;
        CHECK(std::fabs(sum_p - 1.0) < 1e-4, "Alg 11 (Approximation): probability conservation holds");

        // Alg 9 & 12 full pipeline: Transient Multi-Step FSP with Preconditioning
        PreconditionedFspResult r9 = alg9_preconditioned_fsp(net, 0.0, 0.05, 1e-2, 0.01, 4);
        CHECK(r9.converged, "Alg 9 (Preconditioned FSP): full multi-step converged");
        CHECK(!r9.steps.empty(), "Alg 9: recorded transient subinterval steps");

        // Alg 12: evaluate continuous transient solution at an intermediate time
        std::vector<double> P_mid = alg12_transient_solution(r9.steps, r9.full_states, 0.025);
        double sum_mid = 0.0;
        for (double v : P_mid) sum_mid += v;
        CHECK(std::fabs(sum_mid - 1.0) < 1e-3, "Alg 12 (TransientSolution): evaluated continuous solution with sum ~ 1");
    }

    std::printf("%s (%d failures)\n", failures ? "FAILED" : "ALL PASSED", failures);
    return failures ? 1 : 0;
}
