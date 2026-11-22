// Thesis Algorithm 5 (semi-orthogonal eigenbasis by block power iteration), CPU/OpenMP.
//
// Differences from the pseudo-code (mathematically equivalent unless noted):
//  * Pivot + scaling is  V <- V * inv(V[I,:])  so that V[I,:] = Identity.  While the pivot
//    set is still changing it is chosen by Gauss-Jordan on the columns (the RREF of the
//    thesis); once it repeats twice it is frozen and only the d x d block is inverted.
//    If the frozen block becomes ill-conditioned (entries of V blow up) pivots are re-selected.
//  * Convergence is judged by the invariant-subspace residual max|K B - B Kred|, because
//    the change between iterates can be tiny while the iteration is still far from converged.
//  * Optional extension: scaled Chebyshev filtering in place of plain power steps
//    (opt.cutoff > 0), which needs ~sqrt(sigma/gap) instead of sigma/gap products.
//  * Output includes Kred = (K B)[I,:], so K B = B Kred.

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>

#include "cme/eigenbasis.hpp"
#include "cme/small_dense.hpp"

namespace cme {
namespace {

constexpr int MAXD = 64;

/**
 * @brief Evaluates fused Sparse Matrix-Dense Matrix product: out = a * D * X + b * X + c * Z.
 *
 * Parallelized over rows using OpenMP.
 *
 * @param D Sparse matrix in CSR format (dimension n x n).
 * @param X Dense input matrix of size n x d (row-major).
 * @param out Dense output matrix of size n x d (row-major).
 * @param d Number of columns (basis dimension).
 * @param a Scalar multiplier for sparse product D * X.
 * @param b Scalar multiplier for input X.
 * @param c Scalar multiplier for accumulation matrix Z.
 * @param Z Optional accumulation matrix of size n x d (may alias out).
 */
void spmm(const Csr& D, const double* X, double* out, int d, double a, double b, double c, const double* Z) {
    const int32_t n = D.n;

    // Distribute rows across OpenMP threads
    #pragma omp parallel for schedule(static)
    for (int32_t i = 0; i < n; ++i) {
        // Local register accumulator array for row i
        double acc[MAXD];
        // Initialize accumulator to zero
        for (int q = 0; q < d; ++q) acc[q] = 0.0;

        // Loop over non-zero elements in row i of sparse matrix D
        for (int32_t e = D.row_ptr[i]; e < D.row_ptr[i + 1]; ++e) {
            const double v = D.val[e];
            // Pointer to start of row D.col[e] in dense matrix X
            const double* x = X + static_cast<int64_t>(D.col[e]) * d;
            // Accumulate sparse matrix-vector dot product
            for (int q = 0; q < d; ++q) acc[q] += v * x[q];
        }

        // Output row offset in flat array
        const int64_t o = static_cast<int64_t>(i) * d;
        // Compute linear combination: r = a * (D*X)_i + b * X_i + c * Z_i
        for (int q = 0; q < d; ++q) {
            double r = a * acc[q] + b * X[o + q];
            if (Z) r += c * Z[o + q];
            out[o + q] = r;
        }
    }
}

/**
 * @brief Finds row index maximizing absolute value in column j of dense matrix V.
 *
 * @param V Dense matrix of size n x d (row-major).
 * @param n Number of rows.
 * @param d Number of columns.
 * @param j Column index to search.
 * @return Row index with maximum absolute magnitude in column j.
 */
int64_t argmax_abs_col(const double* V, int64_t n, int d, int j) {
    int64_t best = 0;
    double bv = -1.0;

    #pragma omp parallel
    {
        int64_t lb = 0;
        double lv = -1.0;
        // Find local thread maximum
        #pragma omp for nowait
        for (int64_t i = 0; i < n; ++i) {
            double a = std::fabs(V[i * d + j]);
            if (a > lv) { lv = a; lb = i; }
        }
        // Reduce local maximum to global maximum across threads
        #pragma omp critical
        {
            if (lv > bv || (lv == bv && lb < best)) { bv = lv; best = lb; }
        }
    }
    return best;
}

/**
 * @brief Performs column Gauss-Jordan pivoting on dense matrix V so that V[piv[j], :] = e_j.
 *
 * @param V Dense basis matrix of size n x d (row-major), updated in place.
 * @param n Number of rows.
 * @param d Number of columns.
 * @param piv Output vector of length d receiving selected pivot row indices.
 */
void normalise_pivot_search(double* V, int64_t n, int d, std::vector<int64_t>& piv) {
    piv.assign(d, 0);
    double w[MAXD];

    // Iterate through each basis column j
    for (int j = 0; j < d; ++j) {
        // Find row index p with largest magnitude in column j
        int64_t p = argmax_abs_col(V, n, d, j);
        double pv = V[p * d + j];
        // Ensure pivot is not degenerate
        if (std::fabs(pv) < 1e-300) throw std::runtime_error("basis lost rank during pivoting");

        // Record pivot row
        piv[j] = p;
        const double inv = 1.0 / pv;

        // Copy pivot row entries
        for (int c = 0; c < d; ++c) w[c] = V[p * d + c];
        w[j] = 0.0;

        // Eliminate column j in all other rows in parallel
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < n; ++i) {
            double* row = V + i * d;
            const double vj = row[j] * inv;
            row[j] = vj;
            for (int c = 0; c < d; ++c) row[c] -= w[c] * vj;
        }
    }
}

/**
 * @brief Post-multiplies V by inverse of pivot block: V <- V * inv(V[piv, :]).
 *
 * @param V Dense basis matrix of size n x d (row-major).
 * @param n Number of rows.
 * @param d Number of columns.
 * @param piv Fixed pivot row indices of length d.
 * @return Maximum absolute value in V after normalisation.
 */
double normalise_fixed(double* V, int64_t n, int d, const std::vector<int64_t>& piv) {
    // Extract d x d submatrix S = V[piv, :]
    std::vector<double> S(static_cast<size_t>(d) * d);
    for (int j = 0; j < d; ++j) {
        for (int c = 0; c < d; ++c) {
            S[j * d + c] = V[piv[j] * d + c];
        }
    }
    // Invert d x d matrix S in place
    dense_inverse(d, S.data());

    double mx = 0.0;
    // Multiplies every row of V by S^-1 and tracks maximum absolute entry
    #pragma omp parallel for reduction(max : mx) schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        double* row = V + i * d;
        double tmp[MAXD];
        for (int c = 0; c < d; ++c) {
            double s = 0.0;
            for (int k = 0; k < d; ++k) s += row[k] * S[k * d + c];
            tmp[c] = s;
            mx = std::max(mx, std::fabs(s));
        }
        for (int c = 0; c < d; ++c) row[c] = tmp[c];
    }
    return mx;
}

/**
 * @brief Computes invariant subspace residual: max_{i, j} |(K * B)_{ij} - (B * K_{red})_{ij}|.
 *
 * @param KB Evaluated product K * B of size n x d.
 * @param B Current basis matrix of size n x d.
 * @param piv Pivot row indices.
 * @param n Number of rows.
 * @param d Number of columns.
 * @param Kred Output parameter receiving reduced operator K_{red} = (K * B)[piv, :].
 * @return Invariant subspace residual norm.
 */
double compute_residual(const double* KB, const double* B, const std::vector<int64_t>& piv,
                        int64_t n, int d, std::vector<double>& Kred) {
    // Reduced operator Kred = KB[piv, :]
    Kred.resize(static_cast<size_t>(d) * d);
    for (int j = 0; j < d; ++j) {
        for (int c = 0; c < d; ++c) {
            Kred[j * d + c] = KB[piv[j] * d + c];
        }
    }

    double max_res = 0.0;
    // Compute maximum difference |KB - B * Kred|
    #pragma omp parallel for reduction(max : max_res) schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        const double* b = B + i * d;
        const double* kb = KB + i * d;
        for (int c = 0; c < d; ++c) {
            double pred = 0.0;
            for (int k = 0; k < d; ++k) pred += b[k] * Kred[k * d + c];
            max_res = std::max(max_res, std::fabs(kb[c] - pred));
        }
    }
    return max_res;
}

}  // namespace

/**
 * @brief CPU OpenMP implementation of Thesis Algorithm 5.
 *
 * Generates semi-orthogonal basis B of invariant subspace of generator D.
 */
EigenResult eigenbasis_cpu(const Csr& D, double sigma, const EigenOptions& opt) {
    const int64_t n = D.n;
    const int d = opt.d;

    EigenResult res;
    res.n = n;
    res.d = d;

    // Allocate memory for basis matrices V, KV, and workspace
    const size_t sz = static_cast<size_t>(n) * d;
    std::vector<double> V(sz), KV(sz), W(sz);

    // Initialize V with pseudo-random numbers
    std::mt19937_64 rng(opt.seed);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (size_t i = 0; i < sz; ++i) V[i] = dist(rng);

    // Initial pivot search normalisation
    std::vector<int64_t> piv(d);
    normalise_pivot_search(V.data(), n, d, piv);

    // Coefficients for power operator K: K = a * D + b * I
    const double a = 1.0 / sigma;
    const double b = (opt.mode == Mode::Slow) ? 1.0 : 0.0;

    int frozen_count = 0;
    std::vector<int64_t> last_piv = piv;

    double t0 = omp_get_wtime();

    // Main iteration loop
    for (int iter = 0; iter < opt.max_iter; ++iter) {
        res.iterations = iter + 1;

        if (opt.cutoff <= 0.0) {
            // Standard Block Power Iteration: apply K inner times
            for (int s = 0; s < opt.inner; ++s) {
                spmm(D, V.data(), KV.data(), d, a, b, 0.0, nullptr);
                res.spmm_count++;
                if (s + 1 < opt.inner) {
                    std::copy(KV.begin(), KV.end(), V.begin());
                }
            }
        } else {
            // Scaled Chebyshev polynomial filter mode
            // Damps unwanted spectrum outside [0, cutoff]
            const double alpha = opt.cutoff / sigma;
            // Three-term Chebyshev recurrence parameters
            double c_prev = 1.0;
            double c_curr = (2.0 - alpha) / alpha;

            // Step 1: W = K * V
            spmm(D, V.data(), W.data(), d, a, b, 0.0, nullptr);
            res.spmm_count++;

            // Step 2: Recurrence loop
            for (int deg = 1; deg < opt.degree; ++deg) {
                double c_next = 2.0 * ((2.0 - alpha) / alpha) * c_curr - c_prev;
                spmm(D, W.data(), KV.data(), d, (2.0 * a) / alpha, (2.0 * b - 2.0) / alpha + 2.0, -1.0, V.data());
                res.spmm_count++;
                std::copy(W.begin(), W.end(), V.begin());
                std::copy(KV.begin(), KV.end(), W.data());
                c_prev = c_curr;
                c_curr = c_next;
            }
        }

        // Copy latest iterate into V
        std::copy(KV.begin(), KV.end(), V.begin());

        // Pivot selection strategy: freeze pivots once stable
        if (frozen_count >= 2) {
            // Apply fixed normalisation with current pivots
            double mx = normalise_fixed(V.data(), n, d, piv);
            // If condition degrades, unfreeze and re-pivot
            if (mx > 1e6) {
                normalise_pivot_search(V.data(), n, d, piv);
                frozen_count = 0;
            }
        } else {
            // Search for optimal pivot rows
            normalise_pivot_search(V.data(), n, d, piv);
            if (piv == last_piv) {
                frozen_count++;
            } else {
                frozen_count = 0;
                last_piv = piv;
            }
        }

        // Check invariant subspace convergence residual
        if (opt.cutoff > 0.0 || (iter % opt.check_every == 0) || (iter == opt.max_iter - 1)) {
            // Compute action KV = K * V
            spmm(D, V.data(), KV.data(), d, a, b, 0.0, nullptr);
            res.spmm_count++;

            // Evaluate residual norm: max |K B - B Kred|
            res.residual = compute_residual(KV.data(), V.data(), piv, n, d, res.d_reduced);

            // Check if residual satisfied target tolerance
            if (res.residual < opt.tol) {
                res.converged = true;
                break;
            }
        }
    }

    res.seconds_iterate = omp_get_wtime() - t0;
    res.basis = std::move(V);
    res.pivots = piv;

    return res;
}

}  // namespace cme
