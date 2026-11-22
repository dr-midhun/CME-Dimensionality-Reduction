#include "cme/reduction.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include "cme/csr_util.hpp"
#include "cme/small_dense.hpp"

namespace cme {

namespace {

// Matrix multiplication C = A * B for small square matrices (d x d, row-major)
void small_gemm(int d, const double* A, const double* B, double* C) {
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
            double s = 0.0;
            for (int k = 0; k < d; ++k) {
                s += A[i * d + k] * B[k * d + j];
            }
            C[i * d + j] = s;
        }
    }
}

// Matrix-vector multiplication y = A * x (d x d row-major A, length d vectors)
void small_gemv(int d, const double* A, const double* x, double* y) {
    for (int i = 0; i < d; ++i) {
        double s = 0.0;
        for (int j = 0; j < d; ++j) {
            s += A[i * d + j] * x[j];
        }
        y[i] = s;
    }
}

// Computes 1-norm of a d x d matrix (maximum absolute column sum)
double norm1(int d, const double* A) {
    double max_s = 0.0;
    for (int j = 0; j < d; ++j) {
        double s = 0.0;
        for (int i = 0; i < d; ++i) {
            s += std::fabs(A[i * d + j]);
        }
        if (s > max_s) max_s = s;
    }
    return max_s;
}

}  // namespace

// Matrix exponential exp(A) for small d x d matrix via (6,6) Padé approximant with scaling and squaring.
void matrix_exp(int d, const double* A, double* expA) {
    // Scaling: find s >= 0 such that ||A / 2^s||_1 <= 0.5
    double a_norm = norm1(d, A);
    int s = 0;
    if (a_norm > 0.5) {
        s = static_cast<int>(std::ceil(std::log2(a_norm / 0.5)));
        if (s < 0) s = 0;
    }
    double scale = 1.0 / std::ldexp(1.0, s);

    std::vector<double> A_scaled(static_cast<size_t>(d) * d);
    for (size_t i = 0; i < A_scaled.size(); ++i) {
        A_scaled[i] = A[i] * scale;
    }

    // Degree 6 Padé approximation coefficients:
    // c_k = (2p - k)! p! / ( (2p)! k! (p - k)! ) for p = 6
    // c = [1, 1/2, 5/44, 1/66, 1/792, 1/15840, 1/665280]
    // Common factor: multiply all by 665280 for integer numerators:
    // c0 = 665280
    // c1 = 332640
    // c2 = 75600
    // c3 = 10080
    // c4 = 840
    // c5 = 42
    // c6 = 1
    const double c[7] = {
        665280.0, 332640.0, 75600.0, 10080.0, 840.0, 42.0, 1.0
    };

    const size_t sz = static_cast<size_t>(d) * d;
    std::vector<double> A2(sz), A4(sz), A6(sz);
    small_gemm(d, A_scaled.data(), A_scaled.data(), A2.data());
    small_gemm(d, A2.data(), A2.data(), A4.data());
    small_gemm(d, A4.data(), A2.data(), A6.data());

    // U = A * (c1*I + c3*A^2 + c5*A^4)
    // V = c0*I + c2*A^2 + c4*A^4 + c6*A^6
    std::vector<double> U_inner(sz, 0.0);
    std::vector<double> V(sz, 0.0);
    for (int i = 0; i < d; ++i) {
        U_inner[i * d + i] += c[1];
        V[i * d + i] += c[0];
    }
    for (size_t i = 0; i < sz; ++i) {
        U_inner[i] += c[3] * A2[i] + c[5] * A4[i];
        V[i] += c[2] * A2[i] + c[4] * A4[i] + c[6] * A6[i];
    }
    std::vector<double> U(sz);
    small_gemm(d, A_scaled.data(), U_inner.data(), U.data());

    // N_mat = V + U,  D_mat = V - U
    std::vector<double> N_mat(sz), D_mat(sz);
    for (size_t i = 0; i < sz; ++i) {
        N_mat[i] = V[i] + U[i];
        D_mat[i] = V[i] - U[i];
    }

    // Solve D_mat * R = N_mat  =>  R = inv(D_mat) * N_mat
    dense_inverse(d, D_mat.data());
    std::vector<double> R(sz);
    small_gemm(d, D_mat.data(), N_mat.data(), R.data());

    // Squaring phase: R = R^(2^s)
    std::vector<double> temp(sz);
    for (int step = 0; step < s; ++step) {
        small_gemm(d, R.data(), R.data(), temp.data());
        R = temp;
    }

    for (size_t i = 0; i < sz; ++i) {
        expA[i] = R[i];
    }
}

ReductionModel reduce_cme(const Csr& D, const std::vector<double>& P0, const ReductionOptions& opt) {
    if (D.n <= 0) throw std::invalid_argument("D is empty");
    if (opt.d <= 0 || opt.d > D.n) throw std::invalid_argument("invalid basis dimension d");
    if (static_cast<int64_t>(P0.size()) != D.n) throw std::invalid_argument("P0 size does not match D dimension");

    const int64_t n = D.n;
    const int d = opt.d;

    ReductionModel model;
    model.n = n;
    model.d = d;
    model.transform = opt.transform;

    // 1. Compute Right Eigenbasis Mnd (Algorithm 5 on D)
    EigenOptions opt_right;
    opt_right.d = d;
    opt_right.mode = Mode::Slow;
    opt_right.tol = opt.tol;
    opt_right.max_iter = opt.max_iter;
    opt_right.cutoff = opt.cutoff;
    opt_right.degree = opt.degree;
    opt_right.verbose = opt.verbose;
    EigenResult res_right = compute_eigenbasis(D, opt.backend, opt_right);
    if (!res_right.converged) {
        throw std::runtime_error("Right eigenbasis computation did not converge");
    }

    // 2. Compute Left Eigenbasis Tnd (Algorithm 5 on D^T)
    Csr Dt = transpose_csr(D);
    EigenOptions opt_left = opt_right;
    EigenResult res_left = compute_eigenbasis(Dt, opt.backend, opt_left);
    if (!res_left.converged) {
        throw std::runtime_error("Left eigenbasis computation did not converge");
    }

    // Decide permutation pivots I based on transform
    // If transform == LeftEigenbasis: pivots are from Tnd (res_left.pivots)
    // If transform == None: pivots are from Mnd (res_right.pivots)
    const std::vector<int64_t>& primary_pivots = (opt.transform == ReductionTransform::LeftEigenbasis)
                                                    ? res_left.pivots
                                                    : res_right.pivots;

    // Build full permutation vector I
    std::vector<int64_t> perm(n);
    std::vector<bool> in_pivots(n, false);
    for (int j = 0; j < d; ++j) {
        perm[j] = primary_pivots[j];
        in_pivots[primary_pivots[j]] = true;
    }
    int64_t curr = d;
    for (int64_t i = 0; i < n; ++i) {
        if (!in_pivots[i]) {
            perm[curr++] = i;
        }
    }
    std::vector<int64_t> perm_inv(n);
    for (int64_t i = 0; i < n; ++i) {
        perm_inv[perm[i]] = i;
    }

    model.perm = perm;
    model.perm_inv = perm_inv;

    // Re-index Mnd, Tnd, and P0
    model.Mnd_re.resize(static_cast<size_t>(n) * d);
    model.Tnd_re.resize(static_cast<size_t>(n) * d);
    std::vector<double> P0_re(n);

    for (int64_t i = 0; i < n; ++i) {
        int64_t orig = perm[i];
        P0_re[i] = P0[orig];
        for (int c = 0; c < d; ++c) {
            model.Mnd_re[i * d + c] = res_right.basis[orig * d + c];
            model.Tnd_re[i * d + c] = res_left.basis[orig * d + c];
        }
    }

    // Re-scale Tnd_re so its top d x d block is Identity (in case primary pivots came from Mnd)
    {
        std::vector<double> T_top(static_cast<size_t>(d) * d);
        for (int r = 0; r < d; ++r) {
            for (int c = 0; c < d; ++c) {
                T_top[r * d + c] = model.Tnd_re[r * d + c];
            }
        }
        dense_inverse(d, T_top.data());
        std::vector<double> T_new(static_cast<size_t>(n) * d);
        for (int64_t r = 0; r < n; ++r) {
            for (int c = 0; c < d; ++c) {
                double s = 0.0;
                for (int k = 0; k < d; ++k) {
                    s += model.Tnd_re[r * d + k] * T_top[k * d + c];
                }
                T_new[r * d + c] = s;
            }
        }
        model.Tnd_re = std::move(T_new);
    }

    // Re-scale Mnd_re so its top d x d block is Identity (in case primary pivots came from Tnd)
    {
        std::vector<double> M_top(static_cast<size_t>(d) * d);
        for (int r = 0; r < d; ++r) {
            for (int c = 0; c < d; ++c) {
                M_top[r * d + c] = model.Mnd_re[r * d + c];
            }
        }
        dense_inverse(d, M_top.data());
        std::vector<double> M_new(static_cast<size_t>(n) * d);
        for (int64_t r = 0; r < n; ++r) {
            for (int c = 0; c < d; ++c) {
                double s = 0.0;
                for (int k = 0; k < d; ++k) {
                    s += model.Mnd_re[r * d + k] * M_top[k * d + c];
                }
                M_new[r * d + c] = s;
            }
        }
        model.Mnd_re = std::move(M_new);
    }

    const size_t dd = static_cast<size_t>(d) * d;
    model.Q.assign(dd, 0.0);
    model.X0.assign(d, 0.0);

    if (opt.transform == ReductionTransform::None) {
        // --- CASE 1: No Transformation (T = I) ---
        // Thesis eq. 4.64: X(0) = (Tnd^T Mnd)^-1 * Tnd^T * P0
        std::vector<double> TM(dd, 0.0);
        for (int r = 0; r < d; ++r) {
            for (int c = 0; c < d; ++c) {
                double s = 0.0;
                for (int64_t k = 0; k < n; ++k) {
                    s += model.Tnd_re[k * d + r] * model.Mnd_re[k * d + c];
                }
                TM[r * d + c] = s;
            }
        }
        dense_inverse(d, TM.data());

        // TP0 = Tnd^T * P0
        std::vector<double> TP0(d, 0.0);
        for (int r = 0; r < d; ++r) {
            double s = 0.0;
            for (int64_t k = 0; k < n; ++k) {
                s += model.Tnd_re[k * d + r] * P0_re[k];
            }
            TP0[r] = s;
        }
        small_gemv(d, TM.data(), TP0.data(), model.X0.data());

        // Reduced transition matrix Qdd = D_re[0:d, :] * Mnd_re
        // Since D_re is sparse: D_re[r, c] = D[perm[r], perm[c]]
        for (int r = 0; r < d; ++r) {
            int64_t orig_r = perm[r];
            for (int32_t p = D.row_ptr[orig_r]; p < D.row_ptr[orig_r + 1]; ++p) {
                int64_t orig_c = D.col[p];
                int64_t re_c = perm_inv[orig_c];
                double val = D.val[p];
                for (int c = 0; c < d; ++c) {
                    model.Q[r * d + c] += val * model.Mnd_re[re_c * d + c];
                }
            }
        }
    } else {
        // --- CASE 2: Left Eigenbasis Transformation ---
        // Thesis eq. 4.94: X(0) = Tnd^T * P0
        for (int r = 0; r < d; ++r) {
            double s = 0.0;
            for (int64_t k = 0; k < n; ++k) {
                s += model.Tnd_re[k * d + r] * P0_re[k];
            }
            model.X0[r] = s;
        }

        // Reduced transition matrix Qdd = \hat{D}_{dd} = Tnd_re^T * D_re[:, 0:d]
        // (Tnd_re^T * D_re)[:, c] for c in 0..d-1
        // Note: (Tnd_re^T * D_re)[r, c] = sum_k Tnd_re[k, r] * D_re[k, c]
        // But D_re[k, c] = D[perm[k], perm[c]].
        // So for column c, orig_c = perm[c], we iterate over column orig_c of D:
        // which are rows in Dt! Dt.row_ptr[orig_c] contains (orig_k, val).
        for (int c = 0; c < d; ++c) {
            int64_t orig_c = perm[c];
            for (int32_t p = Dt.row_ptr[orig_c]; p < Dt.row_ptr[orig_c + 1]; ++p) {
                int64_t orig_k = Dt.col[p];
                int64_t re_k = perm_inv[orig_k];
                double val = Dt.val[p];
                for (int r = 0; r < d; ++r) {
                    model.Q[r * d + c] += model.Tnd_re[re_k * d + r] * val;
                }
            }
        }

        // Compute \hat{M}_{nd} = RREF(T * Mnd) (thesis eq. 4.97)
        // T is block [I_d, Tnd_re[d:, :]^T; 0, I_m]
        // So for row r < d: (T * Mnd)[r, :] = Mnd_re[r, :] + sum_{m=d}^{n-1} Tnd_re[m, r] * Mnd_re[m, :]
        // For row r >= d:   (T * Mnd)[r, :] = Mnd_re[r, :]
        std::vector<double> TM(static_cast<size_t>(n) * d);
        for (int r = 0; r < d; ++r) {
            for (int c = 0; c < d; ++c) {
                double s = model.Mnd_re[r * d + c];
                for (int64_t m = d; m < n; ++m) {
                    s += model.Tnd_re[m * d + r] * model.Mnd_re[m * d + c];
                }
                TM[r * d + c] = s;
            }
        }
        for (int64_t r = d; r < n; ++r) {
            for (int c = 0; c < d; ++c) {
                TM[r * d + c] = model.Mnd_re[r * d + c];
            }
        }

        // Normalize top d x d block of TM to Identity: \hat{M} = TM * inv(TM[0:d, :])
        std::vector<double> TM_top(dd);
        for (int r = 0; r < d; ++r) {
            for (int c = 0; c < d; ++c) {
                TM_top[r * d + c] = TM[r * d + c];
            }
        }
        dense_inverse(d, TM_top.data());

        model.Mhat_re.resize(static_cast<size_t>(n) * d);
        for (int64_t r = 0; r < n; ++r) {
            for (int c = 0; c < d; ++c) {
                double s = 0.0;
                for (int k = 0; k < d; ++k) {
                    s += TM[r * d + k] * TM_top[k * d + c];
                }
                model.Mhat_re[r * d + c] = s;
            }
        }
    }

    return model;
}

std::vector<double> ReductionModel::solve_reduced(double t) const {
    std::vector<double> Qt(static_cast<size_t>(d) * d);
    for (size_t i = 0; i < Qt.size(); ++i) {
        Qt[i] = Q[i] * t;
    }
    std::vector<double> expQt(static_cast<size_t>(d) * d);
    matrix_exp(d, Qt.data(), expQt.data());

    std::vector<double> Xt(d);
    small_gemv(d, expQt.data(), X0.data(), Xt.data());
    return Xt;
}

std::vector<double> ReductionModel::reconstruct_solution(const std::vector<double>& X_t) const {
    if (static_cast<int>(X_t.size()) != d) {
        throw std::invalid_argument("X_t dimension does not match d");
    }

    std::vector<double> P_re(n, 0.0);
    if (transform == ReductionTransform::None) {
        // \tilde{P}_re = Mnd_re * X(t)
        for (int64_t r = 0; r < n; ++r) {
            double s = 0.0;
            for (int c = 0; c < d; ++c) {
                s += Mnd_re[r * d + c] * X_t[c];
            }
            P_re[r] = s;
        }
    } else {
        // \tilde{P}_re = T^-1 * \hat{M}_re * X(t)
        // First compute Y = \hat{M}_re * X(t)
        std::vector<double> Y(n, 0.0);
        for (int64_t r = 0; r < n; ++r) {
            double s = 0.0;
            for (int c = 0; c < d; ++c) {
                s += Mhat_re[r * d + c] * X_t[c];
            }
            Y[r] = s;
        }
        // T^-1 = [I_d, -Tnd_re[d:, :]^T ; 0, I_m]
        // For r < d: P_re[r] = Y[r] - sum_{m=d}^{n-1} Tnd_re[m, r] * Y[m]
        // For r >= d: P_re[r] = Y[r]
        for (int r = 0; r < d; ++r) {
            double s = Y[r];
            for (int64_t m = d; m < n; ++m) {
                s -= Tnd_re[m * d + r] * Y[m];
            }
            P_re[r] = s;
        }
        for (int64_t r = d; r < n; ++r) {
            P_re[r] = Y[r];
        }
    }

    // Re-index back to original indexing: P[perm[i]] = P_re[i]
    std::vector<double> P_orig(n);
    for (int64_t i = 0; i < n; ++i) {
        P_orig[perm[i]] = P_re[i];
    }
    return P_orig;
}

}  // namespace cme
