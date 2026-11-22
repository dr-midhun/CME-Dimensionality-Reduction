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

// out = a * D X + b * X + c * Z    (Z optional, may alias out; X must not alias out)
void spmm(const Csr& D, const double* X, double* out, int d, double a, double b, double c, const double* Z) {
    const int32_t n = D.n;
    #pragma omp parallel for schedule(static)
    for (int32_t i = 0; i < n; ++i) {
        double acc[MAXD];
        for (int q = 0; q < d; ++q) acc[q] = 0.0;
        for (int32_t e = D.row_ptr[i]; e < D.row_ptr[i + 1]; ++e) {
            const double v = D.val[e];
            const double* x = X + static_cast<int64_t>(D.col[e]) * d;
            for (int q = 0; q < d; ++q) acc[q] += v * x[q];
        }
        const int64_t o = static_cast<int64_t>(i) * d;
        for (int q = 0; q < d; ++q) {
            double r = a * acc[q] + b * X[o + q];
            if (Z) r += c * Z[o + q];
            out[o + q] = r;
        }
    }
}

int64_t argmax_abs_col(const double* V, int64_t n, int d, int j) {
    int64_t best = 0;
    double bv = -1.0;
    #pragma omp parallel
    {
        int64_t lb = 0;
        double lv = -1.0;
        #pragma omp for nowait
        for (int64_t i = 0; i < n; ++i) {
            double a = std::fabs(V[i * d + j]);
            if (a > lv) { lv = a; lb = i; }
        }
        #pragma omp critical
        {
            if (lv > bv || (lv == bv && lb < best)) { bv = lv; best = lb; }
        }
    }
    return best;
}

// Gauss-Jordan on columns: afterwards V[piv[j],:] = e_j.
void normalise_pivot_search(double* V, int64_t n, int d, std::vector<int64_t>& piv) {
    piv.assign(d, 0);
    double w[MAXD];
    for (int j = 0; j < d; ++j) {
        int64_t p = argmax_abs_col(V, n, d, j);
        double pv = V[p * d + j];
        if (std::fabs(pv) < 1e-300) throw std::runtime_error("basis lost rank during pivoting");
        piv[j] = p;
        const double inv = 1.0 / pv;
        for (int c = 0; c < d; ++c) w[c] = V[p * d + c];
        w[j] = 0.0;
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < n; ++i) {
            double* row = V + i * d;
            const double vj = row[j] * inv;
            row[j] = vj;
            for (int c = 0; c < d; ++c) row[c] -= w[c] * vj;
        }
    }
}

// V <- V * inv(V[piv,:]); returns max|V| afterwards.
double normalise_fixed(double* V, int64_t n, int d, const std::vector<int64_t>& piv) {
    std::vector<double> S(static_cast<size_t>(d) * d);
    for (int j = 0; j < d; ++j)
        for (int c = 0; c < d; ++c) S[j * d + c] = V[piv[j] * d + c];
    dense_inverse(d, S.data());
    double mx = 0.0;
    #pragma omp parallel for reduction(max : mx) schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        double* row = V + i * d;
        double out[MAXD];
        for (int c = 0; c < d; ++c) {
            double a = 0.0;
            for (int m = 0; m < d; ++m) a += row[m] * S[m * d + c];
            out[c] = a;
        }
        for (int c = 0; c < d; ++c) { row[c] = out[c]; mx = std::max(mx, std::fabs(out[c])); }
    }
    return mx;
}

}  // namespace

EigenResult eigenbasis_cpu(const Csr& D, double sigma, const EigenOptions& opt) {
    const int d = opt.d;
    const int64_t n = D.n;
    if (d < 1 || d > MAXD) throw std::runtime_error("d must be in [1, 64]");
    const size_t nd = static_cast<size_t>(n) * d;
    const bool slow = opt.mode == Mode::Slow;
    const bool cheby = opt.cutoff > 0.0;
    const double ka = 1.0 / sigma, kb = slow ? 1.0 : 0.0;   // K X = ka * D X + kb * X

    std::vector<double> V(nd), W(nd), Y0, Y1;
    {
        std::mt19937_64 gen(opt.seed);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        for (auto& x : V) x = dist(gen);
    }
    if (cheby) { Y0.resize(nd); Y1.resize(nd); }

    // Chebyshev filter parameters (damp [lo,hi], normalise the polynomial at r0).
    double lo = 0, hi = 0, r0 = 0, e = 1, ctr = 0, s1 = 1;
    if (cheby) {
        if (slow) { lo = -sigma; hi = -opt.cutoff; r0 = 0.0; }
        else      { lo = -opt.cutoff; hi = 0.0; r0 = -sigma; }
        e = 0.5 * (hi - lo);
        ctr = 0.5 * (hi + lo);
        s1 = e / (r0 - ctr);
    }

    std::vector<int64_t> piv;
    bool frozen = false;
    int stable = 0;
    std::vector<int64_t> piv_prev;
    EigenResult res;
    res.n = n;
    res.d = d;
    const double t0 = omp_get_wtime();
    std::vector<double> Kred(static_cast<size_t>(d) * d);

    auto normalise = [&]() {
        if (!frozen) {
            normalise_pivot_search(V.data(), n, d, piv);
            stable = (piv == piv_prev) ? stable + 1 : 0;
            piv_prev = piv;
            if (stable >= 2) frozen = true;
        } else if (normalise_fixed(V.data(), n, d, piv) > 1e3) {
            frozen = false;  // pivot block went bad: reselect
            stable = 0;
            normalise_pivot_search(V.data(), n, d, piv);
            piv_prev = piv;
        }
    };
    // W = K V; Kred = W[piv,:]; returns max|W - V Kred|
    auto residual = [&]() {
        spmm(D, V.data(), W.data(), d, ka, kb, 0.0, nullptr);
        ++res.spmm_count;
        for (int j = 0; j < d; ++j)
            for (int c = 0; c < d; ++c) Kred[j * d + c] = W[piv[j] * d + c];
        double r = 0.0;
        #pragma omp parallel for reduction(max : r) schedule(static)
        for (int64_t i = 0; i < n; ++i)
            for (int c = 0; c < d; ++c) {
                double a = 0.0;
                for (int m = 0; m < d; ++m) a += V[i * d + m] * Kred[m * d + c];
                r = std::max(r, std::fabs(W[i * d + c] - a));
            }
        return r;
    };

    int it = 0;
    for (; it < opt.max_iter; ++it) {
        normalise();
        if (cheby || it % std::max(1, opt.check_every) == 0) {
            res.residual = residual();
            if (opt.verbose) std::fprintf(stderr, "  iter %6d  residual %.3e%s\n", it, res.residual, frozen ? "  (pivots frozen)" : "");
            if (res.residual < opt.tol) { res.converged = true; break; }
        }
        if (cheby) {
            // Y1 = (s1/e)(D - ctr I) V ;  Y_{k+1} = (2 s_{k+1}/e)(D - ctr I) Y_k - s_{k+1} s_k Y_{k-1}
            double sk = s1;
            spmm(D, V.data(), Y1.data(), d, s1 / e, -s1 * ctr / e, 0.0, nullptr);
            Y0 = V;
            ++res.spmm_count;
            for (int k = 1; k < opt.degree; ++k) {
                const double sn = 1.0 / (2.0 / s1 - sk);
                // Y0 <- (2 sn/e)(D Y1 - ctr Y1) - sn*sk*Y0 ; then swap so Y1 is the newest
                spmm(D, Y1.data(), Y0.data(), d, 2.0 * sn / e, -2.0 * sn * ctr / e, -sn * sk, Y0.data());
                ++res.spmm_count;
                Y0.swap(Y1);
                sk = sn;
            }
            V.swap(Y1);
        } else {
            for (int s = 0; s < opt.inner; ++s) {
                spmm(D, V.data(), W.data(), d, ka, kb, 0.0, nullptr);
                ++res.spmm_count;
                V.swap(W);
            }
        }
    }
    if (!res.converged) {          // leave V normalised and report the final residual
        normalise();
        res.residual = residual();
        res.converged = res.residual < opt.tol;
    }
    res.iterations = it;
    res.seconds_iterate = omp_get_wtime() - t0;
    res.d_reduced = Kred;
    res.pivots = piv;
    res.basis = std::move(V);
    return res;
}

}  // namespace cme
