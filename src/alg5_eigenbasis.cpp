#include "cme/eigenbasis.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#ifdef CME_WITH_LAPACK
extern "C" void dgeev_(const char* jobvl, const char* jobvr, const int* n, double* a, const int* lda,
                       double* wr, double* wi, double* vl, const int* ldvl, double* vr,
                       const int* ldvr, double* work, const int* lwork, int* info);
#endif

namespace cme {

double gershgorin_shift(const Csr& D) {
    double m = 0.0;
    for (int32_t i = 0; i < D.n; ++i)
        for (int32_t e = D.row_ptr[i]; e < D.row_ptr[i + 1]; ++e)
            if (D.col[e] == i) m = std::max(m, std::fabs(D.val[e]));
    // Columns of a CME generator sum to zero, so every eigenvalue lies in a disc
    // |z + a_i| <= a_i with a_i = |D_ii|; hence |lambda| <= 2 max a_i.
    return m > 0.0 ? 2.0 * m : 1.0;
}

bool small_eig(int d, const std::vector<double>& A, std::vector<double>& re, std::vector<double>& im,
               std::vector<double>& vr) {
#ifdef CME_WITH_LAPACK
    // Row-major memory read as column-major is the transpose, which has the same eigenvalues
    // but different eigenvectors, so transpose explicitly.
    std::vector<double> a(static_cast<size_t>(d) * d);
    for (int i = 0; i < d; ++i)
        for (int j = 0; j < d; ++j) a[j * d + i] = A[i * d + j];
    re.assign(d, 0.0);
    im.assign(d, 0.0);
    vr.assign(static_cast<size_t>(d) * d, 0.0);
    int info = 0, lwork = -1, one = 1;
    double wq = 0.0;
    dgeev_("N", "V", &d, a.data(), &d, re.data(), im.data(), nullptr, &one, vr.data(), &d, &wq, &lwork, &info);
    lwork = static_cast<int>(wq);
    std::vector<double> work(lwork);
    dgeev_("N", "V", &d, a.data(), &d, re.data(), im.data(), nullptr, &one, vr.data(), &d, work.data(), &lwork, &info);
    return info == 0;
#else
    (void)d; (void)A; (void)re; (void)im; (void)vr;
    return false;
#endif
}

EigenResult compute_eigenbasis(const Csr& D, Backend backend, const EigenOptions& opt) {
    if (opt.d < 1 || opt.d > 64) throw std::runtime_error("d must be in [1, 64]");
    if (opt.d >= D.n) throw std::runtime_error("d must be smaller than the matrix dimension");
    const double sigma = opt.shift > 0.0 ? opt.shift : gershgorin_shift(D);
    if (opt.cutoff > 0.0 && opt.cutoff >= sigma) throw std::runtime_error("cutoff must be smaller than sigma");

    EigenResult r;
    if (backend == Backend::Cpu) {
        r = eigenbasis_cpu(D, sigma, opt);
    } else {
#ifdef CME_WITH_CUDA
        r = eigenbasis_cuda(D, sigma, opt);
#else
        throw std::runtime_error("built without CUDA support (rebuild with GPU=1 / -DCME_ENABLE_CUDA=ON)");
#endif
    }
    r.shift = sigma;
    // Backends return Kred (restriction of K) in d_reduced; map back to D.
    const int d = r.d;
    for (int i = 0; i < d; ++i)
        for (int j = 0; j < d; ++j) {
            double kv = r.d_reduced[i * d + j];
            if (opt.mode == Mode::Slow && i == j) kv -= 1.0;
            r.d_reduced[i * d + j] = sigma * kv;
        }
    small_eig(d, r.d_reduced, r.eig_re, r.eig_im, r.eigvec_reduced);
    // Sanity check for filter mode: all Ritz values must lie in the wanted region, otherwise sigma
    // probably under-estimates the spectral radius and the filter amplified the unwanted modes.
    if (opt.cutoff > 0.0 && !r.eig_re.empty()) {
        for (int j = 0; j < d; ++j) {
            double mag = std::hypot(r.eig_re[j], r.eig_im[j]);
            bool bad = (opt.mode == Mode::Slow) ? (mag > 1.5 * opt.cutoff) : (mag < 0.5 * opt.cutoff);
            if (bad) {
                r.warning = "Ritz value " + std::to_string(r.eig_re[j]) +
                            " is outside the wanted region: check --cutoff/--d, and that sigma (--shift) is >= the spectral radius of D";
                r.converged = false;
                break;
            }
        }
    }
    return r;
}

}  // namespace cme
