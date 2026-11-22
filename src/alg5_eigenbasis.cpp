#include "cme/eigenbasis.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#ifdef CME_WITH_LAPACK
// External declaration of LAPACK general matrix eigensolver dgeev
extern "C" void dgeev_(const char* jobvl, const char* jobvr, const int* n, double* a, const int* lda,
                       double* wr, double* wi, double* vl, const int* ldvl, double* vr,
                       const int* ldvr, double* work, const int* lwork, int* info);
#endif

namespace cme {

/**
 * @brief Computes Gershgorin upper bound on the spectral radius \rho(D) of transition matrix D.
 *
 * For a CME generator, column sums are zero and diagonal entries are negative outflows.
 * Every eigenvalue lies within the union of Gershgorin discs |z - D_{ii}| \le \sum_{j \neq i} |D_{ji}| = |D_{ii}|.
 * Thus |\lambda| \le 2 \max_i |D_{ii}|.
 *
 * @param D Full infinitesimal generator transition matrix in CSR format.
 * @return Positive upper bound shift parameter \sigma \ge \rho(D).
 */
double gershgorin_shift(const Csr& D) {
    double m = 0.0;
    // Iterate over each row i of matrix D
    for (int32_t i = 0; i < D.n; ++i) {
        // Iterate over non-zero elements in row i
        for (int32_t e = D.row_ptr[i]; e < D.row_ptr[i + 1]; ++e) {
            // Find diagonal entry (row == column)
            if (D.col[e] == i) {
                // Track maximum diagonal magnitude |D_ii|
                m = std::max(m, std::fabs(D.val[e]));
            }
        }
    }
    // Return Gershgorin shift 2 * max |D_ii|, defaulting to 1.0 if matrix is zero
    return m > 0.0 ? 2.0 * m : 1.0;
}

/**
 * @brief Computes eigenvalues and right eigenvectors of a small d x d dense matrix using LAPACK dgeev.
 *
 * @param d Matrix dimension (d \le 64).
 * @param A Input dense matrix (row-major).
 * @param re Output vector for real parts of eigenvalues.
 * @param im Output vector for imaginary parts of eigenvalues.
 * @param vr Output vector for right eigenvectors (column-major LAPACK format).
 * @return True if LAPACK dgeev converged successfully; false otherwise.
 */
bool small_eig(int d, const std::vector<double>& A, std::vector<double>& re, std::vector<double>& im,
               std::vector<double>& vr) {
#ifdef CME_WITH_LAPACK
    // LAPACK expects Fortran column-major storage; transpose row-major matrix A into column-major buffer a
    std::vector<double> a(static_cast<size_t>(d) * d);
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
            a[j * d + i] = A[i * d + j];
        }
    }

    // Allocate eigenvalue output vectors
    re.assign(d, 0.0);
    im.assign(d, 0.0);
    // Allocate eigenvector matrix
    vr.assign(static_cast<size_t>(d) * d, 0.0);

    int info = 0;
    int lwork = -1;
    int one = 1;
    double wq = 0.0;

    // Workspace query: determine optimal work array size
    dgeev_("N", "V", &d, a.data(), &d, re.data(), im.data(), nullptr, &one, vr.data(), &d, &wq, &lwork, &info);
    lwork = static_cast<int>(wq);
    std::vector<double> work(lwork);

    // Compute eigenvalues and right eigenvectors
    dgeev_("N", "V", &d, a.data(), &d, re.data(), im.data(), nullptr, &one, vr.data(), &d, work.data(), &lwork, &info);
    return info == 0;
#else
    // LAPACK disabled: ignore unused parameters and return false
    (void)d; (void)A; (void)re; (void)im; (void)vr;
    return false;
#endif
}

/**
 * @brief Top-level interface for computing semi-orthogonal invariant eigenbasis (Thesis Algorithm 5).
 *
 * Dispatches to CPU (OpenMP) or GPU (CUDA) backend, transforms the reduced spectrum back to original
 * generator D, and checks Ritz values against convergence criteria.
 *
 * @param D Full infinitesimal generator transition matrix in CSR format.
 * @param backend Hardware execution backend (Cpu or Cuda).
 * @param opt Eigenbasis configuration parameters.
 * @return EigenResult holding basis vectors, pivot indices, and reduced operator.
 */
EigenResult compute_eigenbasis(const Csr& D, Backend backend, const EigenOptions& opt) {
    // Validate basis dimension d
    if (opt.d < 1 || opt.d > 64) throw std::runtime_error("d must be in [1, 64]");
    // Ensure basis dimension is smaller than state space dimension
    if (opt.d >= D.n) throw std::runtime_error("d must be smaller than the matrix dimension");

    // Determine Gershgorin shift parameter sigma
    const double sigma = opt.shift > 0.0 ? opt.shift : gershgorin_shift(D);
    // Validate Chebyshev cutoff frequency
    if (opt.cutoff > 0.0 && opt.cutoff >= sigma) throw std::runtime_error("cutoff must be smaller than sigma");

    EigenResult r;
    // Dispatch computation to CPU backend
    if (backend == Backend::Cpu) {
        r = eigenbasis_cpu(D, sigma, opt);
    } else {
#ifdef CME_WITH_CUDA
        // Dispatch computation to CUDA GPU backend
        r = eigenbasis_cuda(D, sigma, opt);
#else
        throw std::runtime_error("built without CUDA support (rebuild with GPU=1 / -DCME_ENABLE_CUDA=ON)");
#endif
    }
    // Record shift parameter used
    r.shift = sigma;

    // Backends return Kred (restriction of power matrix K) in d_reduced; map back to generator D:
    // For Slow mode: K = I + D / sigma  =>  D = sigma * (K - I)
    // For Fast mode: K = D / sigma      =>  D = sigma * K
    const int d = r.d;
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
            double kv = r.d_reduced[i * d + j];
            if (opt.mode == Mode::Slow && i == j) {
                kv -= 1.0;
            }
            r.d_reduced[i * d + j] = sigma * kv;
        }
    }

    // Compute dense eigenvalues of the d x d reduced matrix
    small_eig(d, r.d_reduced, r.eig_re, r.eig_im, r.eigvec_reduced);

    // Sanity check for Chebyshev filtered modes: ensure extracted eigenvalues lie within wanted window
    if (opt.cutoff > 0.0 && !r.eig_re.empty()) {
        for (int j = 0; j < d; ++j) {
            // Magnitude |lambda| = sqrt(re^2 + im^2)
            double mag = std::hypot(r.eig_re[j], r.eig_im[j]);
            // Flag if slow mode Ritz value is beyond cutoff or fast mode is below cutoff
            bool bad = (opt.mode == Mode::Slow) ? (mag > 1.5 * opt.cutoff) : (mag < 0.5 * opt.cutoff);
            if (bad) {
                r.warning = "Ritz value " + std::to_string(r.eig_re[j]) +
                            " is outside the wanted region: check --cutoff/--d, and that sigma (--shift) is >= the spectral radius of D";
                r.converged = false;
                break;
            }
        }
    }

    // Return completed eigenbasis result
    return r;
}

}  // namespace cme
