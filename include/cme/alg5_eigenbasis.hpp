#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "cme/csr.hpp"

namespace cme {

enum class Mode { Slow, Fast };      // slow: eigenvalues of D nearest 0; fast: largest |lambda|
enum class Backend { Cpu, Cuda };

struct EigenOptions {
    int d = 2;                // basis dimension (<= 64)
    Mode mode = Mode::Slow;
    double shift = 0.0;       // sigma; <= 0 means Gershgorin bound 2*max|D_ii|
    double tol = 1e-10;       // stop when max|K B - B Kred| < tol (K = power matrix, spectrum in [0,1])
    int max_iter = 200000;    // outer iterations (each = `inner` power steps or one filter of `degree`)
    int inner = 4;            // plain power mode: SpMM products between normalisations
    int check_every = 10;     // plain power mode: residual check period (filter mode checks every pass)
    // Chebyshev acceleration (optional, real or nearly real spectra). With cutoff > 0 the
    // eigenvalues of D with |lambda| < cutoff (slow) or > cutoff (fast) are wanted, the rest
    // is damped by a scaled Chebyshev polynomial of the given degree (thesis eq. 4.55:
    // cutoff = |ln(eps)| / t_eps).
    double cutoff = 0.0;
    int degree = 100;
    uint64_t seed = 1;
    bool verbose = false;
};

struct EigenResult {
    int64_t n = 0;
    int d = 0;
    double shift = 0.0;
    std::vector<double> basis;      // n x d row-major; basis[pivots[j]*d + c] == (c==j)
    std::vector<int64_t> pivots;    // d pivot rows
    std::vector<double> d_reduced;  // d x d row-major: D restricted to span(basis)
    std::vector<double> eig_re, eig_im;  // eigenvalues of d_reduced (needs LAPACK)
    std::vector<double> eigvec_reduced;  // d x d column-major (LAPACK dgeev VR layout)
    int iterations = 0;
    double residual = 0.0;          // max |K B - B Kred|
    int64_t spmm_count = 0;         // number of sparse x dense products
    bool converged = false;
    std::string warning;            // non-empty if the result looks untrustworthy
    double seconds_iterate = 0.0;
};

// Power matrix (never formed explicitly):
//   Slow: K = I + D/sigma  (eigenvalues of D near 0 become the largest)
//   Fast: K = D/sigma      (largest |lambda| of D stay largest)
// sigma defaults to the Gershgorin bound 2*max|D_ii| (all |lambda| <= sigma).
double gershgorin_shift(const Csr& D);

// Semi-orthogonal eigenbasis (thesis Algorithm 5) of the d dominant
// eigenvalues of D, i.e. a basis B of the dominant invariant subspace with
// B[pivots,:] = I, so  D B = B * d_reduced.
EigenResult compute_eigenbasis(const Csr& D, Backend backend, const EigenOptions& opt);

// Backend entry points. They return Kred (the restriction of K) in d_reduced;
// compute_eigenbasis maps it back to D.
EigenResult eigenbasis_cpu(const Csr& D, double sigma, const EigenOptions& opt);
#ifdef CME_WITH_CUDA
EigenResult eigenbasis_cuda(const Csr& D, double sigma, const EigenOptions& opt);
#endif

// Eigen-decomposition of a small real matrix via LAPACK dgeev (no-op if built without).
bool small_eig(int d, const std::vector<double>& A_rowmajor, std::vector<double>& re,
               std::vector<double>& im, std::vector<double>& vr_colmajor);

}  // namespace cme
