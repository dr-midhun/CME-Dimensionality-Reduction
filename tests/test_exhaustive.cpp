// Comprehensive Exhaustive Unit Tests covering all 12 Algorithms and Edge Cases
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <stdexcept>

#include "cme/alg1_cme.hpp"
#include "cme/alg2_state_space.hpp"
#include "cme/alg3_transition.hpp"
#include "cme/alg4_reduction.hpp"
#include "cme/alg5_eigenbasis.hpp"
#include "cme/alg6_7_8_fsp.hpp"
#include "cme/alg9_10_11_12_preconditioned_fsp.hpp"
#include "cme/csr_util.hpp"
#include "cme/network.hpp"

using namespace cme;
static int total_failures = 0;
static int total_tests = 0;

#define TEST_CHECK(cond, msg) do { \
    ++total_tests; \
    if (!(cond)) { \
        std::printf("  [FAIL] %s (line %d)\n", msg, __LINE__); \
        ++total_failures; \
    } else { \
        std::printf("  [OK]   %s\n", msg); \
    } \
} while (0)

// Helper: build standard MM network (eq. 4.76)
Network make_mm_network() {
    Network net;
    net.species = {"S1", "S2", "S3", "S4"};
    net.R = {{1, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 1, 0}, {0, 1, 0, 1}};
    net.S = {{-1, -1, 1, 0}, {1, 1, -1, 0}, {0, 1, -1, 1}, {0, -1, 1, -1}};
    net.rate = {1000, 500, 5, 1};
    net.C = {{1, 0, 1, 1}, {0, 1, 1, 0}};
    net.x0 = {3, 3, 0, 0};
    return net;
}

// -------------------------------------------------------------
// 1. Network Parsing & Utilities
// -------------------------------------------------------------
void test_network_and_csr_utils() {
    std::puts("\n=== Testing Network Parsing and CSR Utilities ===");
    try {
        parse_network("non_existent_file.net");
        TEST_CHECK(false, "Parsing nonexistent file should throw");
    } catch (const std::exception&) {
        TEST_CHECK(true, "Parsing nonexistent file throws exception");
    }

    Network net = parse_network("examples/michaelis_menten.net");
    TEST_CHECK(net.k() == 4, "Parsed network has 4 species");
    TEST_CHECK(net.h() == 4, "Parsed network has 4 reactions");
    TEST_CHECK(net.x0.size() == 4 && net.x0[0] == 3, "Parsed initial state S1=3");

    // CSR Transpose Unit Test
    Csr A;
    A.n = 3;
    A.row_ptr = {0, 2, 3, 4};
    A.col = {0, 1, 2, 0};
    A.val = {1.0, 2.0, 3.0, 4.0};
    Csr At = transpose_csr(A);
    TEST_CHECK(At.n == 3, "Transpose preserves dimension");
    TEST_CHECK(At.nnz() == 4, "Transpose preserves nnz");
    TEST_CHECK(At.row_ptr[1] - At.row_ptr[0] == 2, "Row 0 of At has 2 entries (from col 0)");
}

// -------------------------------------------------------------
// 2. Algorithm 1: Iterative CME State Space Generation
// -------------------------------------------------------------
void test_alg1_cme() {
    std::puts("\n=== Testing Algorithm 1 (Iterative CME Generation) ===");
    Network net = make_mm_network();
    Alg1Result res = alg1_generate_cme(net);

    TEST_CHECK(res.k == 4, "Species count k = 4");
    TEST_CHECK(res.states.size() == 10, "Discovered exactly 10 states (Thesis eq. 4.77)");
    TEST_CHECK(res.D.n == 10, "Generator dimension is 10 x 10");
    TEST_CHECK(res.D.nnz() == 34, "Generator has 34 stored entries (24 off-diag + 10 diag)");

    // Verify all column sums in D are strictly zero
    std::vector<double> csum(res.D.n, 0.0);
    for (int r = 0; r < res.D.n; ++r) {
        for (int p = res.D.row_ptr[r]; p < res.D.row_ptr[r + 1]; ++p) {
            csum[res.D.col[p]] += res.D.val[p];
        }
    }
    double max_err = 0.0;
    for (double c : csum) max_err = std::max(max_err, std::fabs(c));
    TEST_CHECK(max_err < 1e-12, "Algorithm 1: exact probability conservation sum_i D_ij == 0");

    // Empty network exception
    Network empty_net;
    try {
        alg1_generate_cme(empty_net);
        TEST_CHECK(false, "Empty network should throw exception");
    } catch (const std::invalid_argument&) {
        TEST_CHECK(true, "Empty network throws invalid_argument");
    }
}

// -------------------------------------------------------------
// 3. Algorithm 2 & 3: Mixed-Radix State Space & Matrix
// -------------------------------------------------------------
void test_alg2_and_3() {
    std::puts("\n=== Testing Algorithm 2 & 3 (State Space & Transition Matrix) ===");
    Network net = make_mm_network();
    StateSpace ss = generate_state_space(net);

    TEST_CHECK(ss.n() == 10, "Algorithm 2 generated 10 states");
    TEST_CHECK(ss.k == 4, "Algorithm 2 species count k = 4");

    // Verify encode / decode isomorphism
    bool encode_ok = true;
    for (size_t i = 0; i < ss.keys.size(); ++i) {
        int N[4];
        ss.decode(ss.keys[i], N);
        uint64_t re_encoded = ss.encode(N);
        if (re_encoded != ss.keys[i]) encode_ok = false;
        int64_t idx = ss.find(ss.keys[i]);
        if (idx != static_cast<int64_t>(i)) encode_ok = false;
    }
    TEST_CHECK(encode_ok, "StateSpace encode/decode/find round-trip verified");
    TEST_CHECK(ss.find(99999999ULL) == -1, "StateSpace find returns -1 for missing key");

    // Algorithm 3
    Csr D = build_transition_matrix(net, ss);
    TEST_CHECK(D.n == 10, "Algorithm 3 D matrix dimension = 10");
    TEST_CHECK(D.nnz() == 34, "Algorithm 3 D matrix nnz = 34");

    double max_diag = 0.0;
    for (int i = 0; i < D.n; ++i) {
        for (int p = D.row_ptr[i]; p < D.row_ptr[i + 1]; ++p) {
            if (D.col[p] == i) max_diag = std::max(max_diag, std::fabs(D.val[p]));
        }
    }
    TEST_CHECK(std::fabs(max_diag - 9000.0) < 1e-9, "Max diagonal is 9000 (Thesis eq. 4.79)");
}

// -------------------------------------------------------------
// 4. Algorithm 4 & 5: Eigenbasis & Dimensionality Reduction
// -------------------------------------------------------------
void test_alg4_and_5(Backend backend) {
    std::puts("\n=== Testing Algorithm 4 & 5 (Eigenbasis & Dimensionality Reduction) ===");
    Network net = make_mm_network();
    StateSpace ss = generate_state_space(net);
    Csr D = build_transition_matrix(net, ss);

    // Alg 5: Slow Modes
    EigenOptions opt;
    opt.d = 4;
    opt.tol = 1e-12;
    EigenResult r = compute_eigenbasis(D, backend, opt);
    TEST_CHECK(r.converged, "Algorithm 5 (Slow): Block power iteration converged");
    TEST_CHECK(r.residual < 1e-9, "Algorithm 5: Invariant subspace residual ||D B - B Dred|| < 1e-9");
    TEST_CHECK(r.pivots.size() == 4, "Algorithm 5: exactly 4 pivots selected");

    // Alg 5: Fast Modes
    EigenOptions fo;
    fo.d = 2;
    fo.mode = Mode::Fast;
    fo.tol = 1e-12;
    EigenResult f = compute_eigenbasis(D, backend, fo);
    TEST_CHECK(f.converged, "Algorithm 5 (Fast): Block power iteration converged");

    // Alg 5: Parameter Validation & Edge Cases
    EigenOptions bad_opt = opt;
    bad_opt.d = 0;
    try {
        compute_eigenbasis(D, backend, bad_opt);
        TEST_CHECK(false, "d=0 should throw exception");
    } catch (const std::exception&) {
        TEST_CHECK(true, "d=0 throws runtime_error");
    }
    bad_opt.d = 10;
    try {
        compute_eigenbasis(D, backend, bad_opt);
        TEST_CHECK(false, "d >= D.n should throw exception");
    } catch (const std::exception&) {
        TEST_CHECK(true, "d >= D.n throws runtime_error");
    }

    // Matrix Exponential Unit Test
    std::vector<double> A_test = {0.0, 1.0, -2.0, -3.0}; // eigenvalues -1, -2
    std::vector<double> expA(4);
    matrix_exp(2, A_test.data(), expA.data());
    // Trace and determinant checks: det(exp(A)) = exp(trace(A)) = exp(-3)
    double det = expA[0] * expA[3] - expA[1] * expA[2];
    TEST_CHECK(std::fabs(det - std::exp(-3.0)) < 1e-4, "matrix_exp: det(exp(A)) == exp(tr(A))");

    // Alg 4: LeftEigenbasis vs None
    std::vector<double> P0(10, 0.1);
    ReductionOptions ropt;
    ropt.d = 4;
    ropt.transform = ReductionTransform::LeftEigenbasis;
    ropt.backend = backend;
    ReductionModel red_L = reduce_cme(D, P0, ropt);

    double sum_X0 = 0.0;
    for (double v : red_L.X0) sum_X0 += v;
    TEST_CHECK(std::fabs(sum_X0 - 1.0) < 1e-6, "Algorithm 4 (LeftEigenbasis): sum(X0) == 1");

    // Check colsum(Q) == 0
    for (int c = 0; c < red_L.d; ++c) {
        double col_sum = 0.0;
        for (int row = 0; row < red_L.d; ++row) col_sum += red_L.Q[row * red_L.d + c];
        TEST_CHECK(std::fabs(col_sum) < 1e-6, "Algorithm 4: reduced generator column sum is 0");
    }

    std::vector<double> P_t05 = red_L.evaluate(0.05);
    double sum_P = 0.0;
    for (double v : P_t05) sum_P += v;
    TEST_CHECK(std::fabs(sum_P - 1.0) < 1e-6, "Algorithm 4: reconstructed P(t) sum is 1");
}

// -------------------------------------------------------------
// 5. Algorithms 6, 7, 8: Finite State Projection
// -------------------------------------------------------------
void test_alg6_7_8() {
    std::puts("\n=== Testing Algorithms 6, 7, 8 (Finite State Projection) ===");
    Network net = make_mm_network();
    FspOptions opt;
    opt.tf = 0.05;
    opt.eps = 1e-2;

    // Algorithm 6
    FspResult r6 = alg6_fsp(net, opt);
    TEST_CHECK(r6.converged, "Algorithm 6 converged");
    TEST_CHECK(r6.error_bound <= opt.eps, "Algorithm 6 error within eps");
    TEST_CHECK(r6.prob.size() == r6.states.size(), "Algorithm 6 distribution matches state count");

    // Algorithm 7
    opt.n_intervals = 5;
    FspResult r7 = alg7_multistep_fixed_fsp(net, opt);
    TEST_CHECK(r7.converged, "Algorithm 7 converged");
    TEST_CHECK(r7.steps == 5, "Algorithm 7 executed exactly 5 intervals");
    TEST_CHECK(r7.error_bound <= opt.eps * 1.05, "Algorithm 7 error bounded");

    // Algorithm 8
    FspResult r8 = alg8_multistep_variable_fsp(net, opt);
    TEST_CHECK(r8.converged, "Algorithm 8 converged");
    TEST_CHECK(r8.steps >= 1, "Algorithm 8 took adaptive steps");
    TEST_CHECK(r8.error_bound <= opt.eps * 1.05, "Algorithm 8 error bounded");
}

// -------------------------------------------------------------
// 6. Algorithms 9, 10, 11, 12: Preconditioned FSP
// -------------------------------------------------------------
void test_alg9_10_11_12() {
    std::puts("\n=== Testing Algorithms 9, 10, 11, 12 (Preconditioned FSP) ===");
    Network net = make_mm_network();
    StateSpace ss = generate_state_space(net);
    Csr D = build_transition_matrix(net, ss);

    std::vector<double> P0(10, 0.0);
    P0[0] = 1.0;

    // Algorithm 10
    PreconditionResult precon = alg10_precondition(D, P0, 4, 1e-3, 0.01);
    TEST_CHECK(precon.tau > 0.0, "Algorithm 10: determined positive step tau");
    TEST_CHECK(precon.d == 4, "Algorithm 10: reduced dimension = 4");
    TEST_CHECK(precon.Z0.size() == 4, "Algorithm 10: Z0 dimension = 4");

    // Algorithm 11
    ApproximationResult approx = alg11_approximation(D, precon, precon.Z_tau);
    double sum_approx = 0.0;
    for (double v : approx.P_next) sum_approx += v;
    TEST_CHECK(std::fabs(sum_approx - 1.0) < 1e-4, "Algorithm 11: back-projection sum is 1");

    // Algorithm 9 & 12
    PreconditionedFspResult r9 = alg9_preconditioned_fsp(net, 0.0, 0.05, 1e-2, 0.01, 4);
    TEST_CHECK(r9.converged, "Algorithm 9: multi-step preconditioned FSP converged");
    TEST_CHECK(!r9.steps.empty(), "Algorithm 9: recorded step history");

    std::vector<double> P_mid = alg12_transient_solution(r9.steps, r9.full_states, 0.025);
    double sum_mid = 0.0;
    for (double v : P_mid) sum_mid += v;
    TEST_CHECK(std::fabs(sum_mid - 1.0) < 0.015, "Algorithm 12: continuous solution sum ~ 1");
}

// -------------------------------------------------------------
// 7. Full Integration Test: End-to-End Execution Flow
// -------------------------------------------------------------
void test_end_to_end_integration() {
    std::puts("\n=== Testing End-to-End CME Integration Flow ===");
    Network net = parse_network("examples/michaelis_menten.net");

    // 1. Generation
    StateSpace ss = generate_state_space(net);
    Csr D = build_transition_matrix(net, ss);
    TEST_CHECK(ss.n() == 10 && D.nnz() == 34, "End-to-End: State Space & Matrix D created");

    // 2. Initial state
    std::vector<double> P0(ss.n(), 0.0);
    P0[0] = 1.0;

    // 3. Dimensionality reduction
    ReductionOptions ropt;
    ropt.d = 4;
    ropt.transform = ReductionTransform::LeftEigenbasis;
    ReductionModel red = reduce_cme(D, P0, ropt);

    // 4. Solve across continuous time grid
    std::vector<double> time_points = {0.001, 0.005, 0.01, 0.02, 0.05};
    bool all_valid = true;
    for (double t : time_points) {
        std::vector<double> Pt = red.evaluate(t);
        double s = 0.0;
        for (double p : Pt) s += p;
        if (std::fabs(s - 1.0) > 1e-5) all_valid = false;
    }
    TEST_CHECK(all_valid, "End-to-End: Continuous trajectory evaluated across time grid with conservation");
}

int main(int argc, char** argv) {
    Backend backend = (argc > 1 && std::string(argv[1]) == "gpu") ? Backend::Cuda : Backend::Cpu;
    std::printf("Running Exhaustive Unit and Integration Tests (Backend: %s)...\n",
                (backend == Backend::Cuda) ? "CUDA" : "CPU");

    test_network_and_csr_utils();
    test_alg1_cme();
    test_alg2_and_3();
    test_alg4_and_5(backend);
    test_alg6_7_8();
    test_alg9_10_11_12();
    test_end_to_end_integration();

    std::puts("\n=======================================================");
    std::printf("Summary: %d / %d assertions passed (%d failures)\n",
                total_tests - total_failures, total_tests, total_failures);
    std::puts("=======================================================");

    return total_failures ? 1 : 0;
}
