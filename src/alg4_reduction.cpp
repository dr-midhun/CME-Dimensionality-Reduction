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

/**
 * @brief Computes matrix-matrix product C = A * B for small dense d x d matrices.
 *
 * @param d Matrix row and column dimension.
 * @param A Pointer to row-major d x d matrix A.
 * @param B Pointer to row-major d x d matrix B.
 * @param C Pointer to output row-major d x d matrix C.
 */
void small_gemm(int d, const double* A, const double* B, double* C) {
    // Loop over each row i of output matrix C
    for (int i = 0; i < d; ++i) {
        // Loop over each column j of output matrix C
        for (int j = 0; j < d; ++j) {
            // Accumulator for inner dot product between row i of A and column j of B
            double s = 0.0;
            // Iterate over contraction dimension k
            for (int k = 0; k < d; ++k) {
                // Multiply A[i, k] by B[k, j] and accumulate
                s += A[i * d + k] * B[k * d + j];
            }
            // Assign computed sum to C[i, j]
            C[i * d + j] = s;
        }
    }
}

/**
 * @brief Computes matrix-vector product y = A * x for small dense d x d matrix A.
 *
 * @param d Dimension of vectors and matrix.
 * @param A Pointer to row-major d x d matrix A.
 * @param x Pointer to input vector x of length d.
 * @param y Pointer to output vector y of length d.
 */
void small_gemv(int d, const double* A, const double* x, double* y) {
    // Loop over each row i of matrix A
    for (int i = 0; i < d; ++i) {
        // Accumulator for dot product between row i and vector x
        double s = 0.0;
        // Loop over column index j
        for (int j = 0; j < d; ++j) {
            // Multiply matrix entry A[i, j] by vector component x[j]
            s += A[i * d + j] * x[j];
        }
        // Store resulting value in output vector element y[i]
        y[i] = s;
    }
}

/**
 * @brief Computes matrix 1-norm (maximum absolute column sum) for a d x d matrix.
 *
 * Formula: ||A||_1 = max_j \sum_i |A_{i, j}|.
 *
 * @param d Matrix dimension.
 * @param A Pointer to row-major d x d matrix A.
 * @return Maximum 1-norm column sum as a double.
 */
double norm1(int d, const double* A) {
    // Maximum column sum found so far
    double max_s = 0.0;
    // Iterate over each column index j
    for (int j = 0; j < d; ++j) {
        // Column sum accumulator for column j
        double s = 0.0;
        // Iterate over each row index i
        for (int i = 0; i < d; ++i) {
            // Add absolute value of matrix entry A[i, j]
            s += std::fabs(A[i * d + j]);
        }
        // Track the maximum column sum across all columns
        if (s > max_s) max_s = s;
    }
    // Return the matrix 1-norm
    return max_s;
}

}  // namespace

/**
 * @brief Computes matrix exponential exp(A) for small d x d matrices using (6,6) Padé approximation with scaling and squaring.
 *
 * @param d Matrix dimension.
 * @param A Input d x d matrix (row-major).
 * @param expA Output d x d matrix representing exp(A) (row-major).
 */
void matrix_exp(int d, const double* A, double* expA) {
    // Calculate 1-norm of input matrix A
    double a_norm = norm1(d, A);
    // Determine scaling power s such that ||A / 2^s||_1 <= 0.5
    int s = 0;
    if (a_norm > 0.5) {
        // Compute integer exponent s = ceil(log2(norm / 0.5))
        s = static_cast<int>(std::ceil(std::log2(a_norm / 0.5)));
        // Ensure non-negative scaling factor
        if (s < 0) s = 0;
    }
    // Scaling multiplier 2^(-s)
    double scale = 1.0 / std::ldexp(1.0, s);

    // Scale input matrix: A_scaled = A * 2^(-s)
    std::vector<double> A_scaled(static_cast<size_t>(d) * d);
    for (size_t i = 0; i < A_scaled.size(); ++i) {
        // Multiply each entry by scaling factor
        A_scaled[i] = A[i] * scale;
    }

    // Degree (6,6) Padé approximation coefficients scaled to integers:
    // c_k = (2p - k)! p! / ((2p)! k! (p - k)!) * 665280
    const double c[7] = {
        665280.0, 332640.0, 75600.0, 10080.0, 840.0, 42.0, 1.0
    };

    // Total elements in d x d matrix
    const size_t sz = static_cast<size_t>(d) * d;
    // Allocate matrix powers A^2, A^4, and A^6
    std::vector<double> A2(sz), A4(sz), A6(sz);
    // A^2 = A_scaled * A_scaled
    small_gemm(d, A_scaled.data(), A_scaled.data(), A2.data());
    // A^4 = A^2 * A^2
    small_gemm(d, A2.data(), A2.data(), A4.data());
    // A^6 = A^4 * A^2
    small_gemm(d, A4.data(), A2.data(), A6.data());

    // Padé numerator U = A * (c1*I + c3*A^2 + c5*A^4)
    // Padé denominator V = c0*I + c2*A^2 + c4*A^4 + c6*A^6
    std::vector<double> U_inner(sz, 0.0);
    std::vector<double> V(sz, 0.0);
    // Initialize diagonal components with identity coefficients
    for (int i = 0; i < d; ++i) {
        U_inner[i * d + i] += c[1];
        V[i * d + i] += c[0];
    }
    // Add contributions of powers of A to U_inner and V
    for (size_t i = 0; i < sz; ++i) {
        U_inner[i] += c[3] * A2[i] + c[5] * A4[i];
        V[i] += c[2] * A2[i] + c[4] * A4[i] + c[6] * A6[i];
    }
    // Multiply U = A_scaled * U_inner
    std::vector<double> U(sz);
    small_gemm(d, A_scaled.data(), U_inner.data(), U.data());

    // Form Padé polynomial matrices: N_mat = V + U,  D_mat = V - U
    std::vector<double> N_mat(sz), D_mat(sz);
    for (size_t i = 0; i < sz; ++i) {
        N_mat[i] = V[i] + U[i];
        D_mat[i] = V[i] - U[i];
    }

    // Invert denominator matrix D_mat in place
    dense_inverse(d, D_mat.data());
    // Solve rational Padé approximant R = inv(D_mat) * N_mat
    std::vector<double> R(sz);
    small_gemm(d, D_mat.data(), N_mat.data(), R.data());

    // Squaring step: repeated self-multiplication s times: R = R^(2^s)
    std::vector<double> temp(sz);
    for (int step = 0; step < s; ++step) {
        // temp = R * R
        small_gemm(d, R.data(), R.data(), temp.data());
        // Update R with squared matrix
        R = temp;
    }

    // Copy computed matrix exponential into output buffer expA
    for (size_t i = 0; i < sz; ++i) {
        expA[i] = R[i];
    }
}

/**
 * @brief Implements CME Dimensionality Reduction (Thesis Algorithm 4).
 *
 * Constructs reduced order generator Q and projection vectors from full matrix D and initial state P0.
 */
ReductionModel reduce_cme(const Csr& D, const std::vector<double>& P0, const ReductionOptions& opt) {
    // Validate that transition matrix D has non-empty state space
    if (D.n <= 0) throw std::invalid_argument("D is empty");
    // Validate that reduced basis dimension d is within valid range [1, D.n]
    if (opt.d <= 0 || opt.d > D.n) throw std::invalid_argument("invalid basis dimension d");
    // Validate that initial distribution size matches matrix dimensions
    if (static_cast<int64_t>(P0.size()) != D.n) throw std::invalid_argument("P0 size does not match D dimension");

    // Full state space dimension n and reduced dimension d
    const int64_t n = D.n;
    const int d = opt.d;

    // Initialize result model structure
    ReductionModel model;
    model.n = n;
    model.d = d;
    model.transform = opt.transform;

    // Step 1: Compute Right Eigenbasis Mnd of D for slowest decaying modes (Thesis Algorithm 5)
    EigenOptions opt_right;
    opt_right.d = d;
    opt_right.mode = Mode::Slow;
    opt_right.tol = opt.tol;
    opt_right.max_iter = opt.max_iter;
    opt_right.cutoff = opt.cutoff;
    opt_right.degree = opt.degree;
    opt_right.verbose = opt.verbose;
    // Call eigenbasis solver to compute right invariant subspace
    EigenResult res_right = compute_eigenbasis(D, opt.backend, opt_right);
    // Check convergence of right eigenbasis
    if (!res_right.converged) {
        throw std::runtime_error("Right eigenbasis computation did not converge");
    }

    // Step 2: Compute Left Eigenbasis Tnd of D^T for slowest decaying modes (Thesis Algorithm 5)
    Csr Dt = transpose_csr(D);
    EigenOptions opt_left = opt_right;
    // Call eigenbasis solver to compute left invariant subspace
    EigenResult res_left = compute_eigenbasis(Dt, opt.backend, opt_left);
    // Check convergence of left eigenbasis
    if (!res_left.converged) {
        throw std::runtime_error("Left eigenbasis computation did not converge");
    }

    // Select primary pivot states:
    // LeftEigenbasis transform uses pivots of Tnd (left eigenbasis)
    // NoTransform (None) uses pivots of Mnd (right eigenbasis)
    const std::vector<int64_t>& primary_pivots = (opt.transform == ReductionTransform::LeftEigenbasis)
                                                    ? res_left.pivots
                                                    : res_right.pivots;

    // Construct full permutation vector I: places pivot states in first d positions [0 .. d-1]
    std::vector<int64_t> perm(n);
    std::vector<bool> in_pivots(n, false);
    for (int j = 0; j < d; ++j) {
        // Place pivot state j into position j
        perm[j] = primary_pivots[j];
        // Mark state as included in pivots
        in_pivots[primary_pivots[j]] = true;
    }
    // Append remaining non-pivot states in positions [d .. n-1]
    int64_t curr = d;
    for (int64_t i = 0; i < n; ++i) {
        if (!in_pivots[i]) {
            perm[curr++] = i;
        }
    }
    // Invert permutation vector for O(1) index lookup: perm_inv[orig] = re_indexed
    std::vector<int64_t> perm_inv(n);
    for (int64_t i = 0; i < n; ++i) {
        perm_inv[perm[i]] = i;
    }

    // Assign permutations to model
    model.perm = perm;
    model.perm_inv = perm_inv;

    // Allocate and re-index basis matrices Mnd, Tnd, and initial distribution P0
    model.Mnd_re.resize(static_cast<size_t>(n) * d);
    model.Tnd_re.resize(static_cast<size_t>(n) * d);
    std::vector<double> P0_re(n);

    // Permute rows of Mnd, Tnd, and elements of P0 into re-indexed coordinate system
    for (int64_t i = 0; i < n; ++i) {
        int64_t orig = perm[i];
        P0_re[i] = P0[orig];
        for (int c = 0; c < d; ++c) {
            model.Mnd_re[i * d + c] = res_right.basis[orig * d + c];
            model.Tnd_re[i * d + c] = res_left.basis[orig * d + c];
        }
    }

    // Normalize Tnd_re so its top d x d submatrix is the Identity matrix
    {
        std::vector<double> T_top(static_cast<size_t>(d) * d);
        for (int r = 0; r < d; ++r) {
            for (int c = 0; c < d; ++c) {
                T_top[r * d + c] = model.Tnd_re[r * d + c];
            }
        }
        // Invert top block
        dense_inverse(d, T_top.data());
        std::vector<double> T_new(static_cast<size_t>(n) * d);
        // Multiply Tnd_re by inverse of top block
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

    // Normalize Mnd_re so its top d x d submatrix is the Identity matrix
    {
        std::vector<double> M_top(static_cast<size_t>(d) * d);
        for (int r = 0; r < d; ++r) {
            for (int c = 0; c < d; ++c) {
                M_top[r * d + c] = model.Mnd_re[r * d + c];
            }
        }
        // Invert top block
        dense_inverse(d, M_top.data());
        std::vector<double> M_new(static_cast<size_t>(n) * d);
        // Multiply Mnd_re by inverse of top block
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

    // Allocate memory for reduced transition rate matrix Q and initial reduced state X0
    const size_t dd = static_cast<size_t>(d) * d;
    model.Q.assign(dd, 0.0);
    model.X0.assign(d, 0.0);

    if (opt.transform == ReductionTransform::None) {
        // --- CASE 1: No Transformation (T = I) ---
        // Thesis eq. 4.64: X(0) = (Tnd^T Mnd)^-1 * Tnd^T * P0
        std::vector<double> TM(dd, 0.0);
        // Compute matrix product TM = Tnd^T * Mnd
        for (int r = 0; r < d; ++r) {
            for (int c = 0; c < d; ++c) {
                double s = 0.0;
                for (int64_t k = 0; k < n; ++k) {
                    s += model.Tnd_re[k * d + r] * model.Mnd_re[k * d + c];
                }
                TM[r * d + c] = s;
            }
        }
        // Invert TM matrix
        dense_inverse(d, TM.data());

        // Calculate vector product TP0 = Tnd^T * P0
        std::vector<double> TP0(d, 0.0);
        for (int r = 0; r < d; ++r) {
            double s = 0.0;
            for (int64_t k = 0; k < n; ++k) {
                s += model.Tnd_re[k * d + r] * P0_re[k];
            }
            TP0[r] = s;
        }
        // Form reduced initial state: X(0) = TM_inv * TP0
        small_gemv(d, TM.data(), TP0.data(), model.X0.data());

        // Form reduced transition rate matrix: Q_dd = D_re[0:d, :] * Mnd_re
        // Exploit sparsity of D
        for (int r = 0; r < d; ++r) {
            int64_t orig_r = perm[r];
            // Iterate over non-zero elements in row orig_r of matrix D
            for (int32_t p = D.row_ptr[orig_r]; p < D.row_ptr[orig_r + 1]; ++p) {
                int64_t orig_c = D.col[p];
                int64_t re_c = perm_inv[orig_c];
                double val = D.val[p];
                // Accumulate sparse product into dense Q
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
            // Project initial probability distribution into X(0)
            model.X0[r] = s;
        }

        // Reduced transition matrix Q_dd = \hat{D}_{dd} = Tnd_re^T * D_re[:, 0:d] (Thesis eq. 4.96)
        // Access column orig_c of D by inspecting row orig_c of transpose matrix D^T
        for (int c = 0; c < d; ++c) {
            int64_t orig_c = perm[c];
            // Iterate over row orig_c in transposed matrix Dt
            for (int32_t p = Dt.row_ptr[orig_c]; p < Dt.row_ptr[orig_c + 1]; ++p) {
                int64_t orig_k = Dt.col[p];
                int64_t re_k = perm_inv[orig_k];
                double val = Dt.val[p];
                // Multiply transpose entry by left eigenbasis element
                for (int r = 0; r < d; ++r) {
                    model.Q[r * d + c] += model.Tnd_re[re_k * d + r] * val;
                }
            }
        }

        // Compute \hat{M}_{nd} = RREF(T * Mnd) (Thesis eq. 4.97)
        // Matrix T is block upper-triangular: [I_d, Tnd_re[d:, :]^T; 0, I_m]
        std::vector<double> TM(static_cast<size_t>(n) * d);
        // For upper rows r < d: (T * Mnd)[r, :] = Mnd_re[r, :] + sum_{m=d}^{n-1} Tnd_re[m, r] * Mnd_re[m, :]
        for (int r = 0; r < d; ++r) {
            for (int c = 0; c < d; ++c) {
                double s = model.Mnd_re[r * d + c];
                for (int64_t m = d; m < n; ++m) {
                    s += model.Tnd_re[m * d + r] * model.Mnd_re[m * d + c];
                }
                TM[r * d + c] = s;
            }
        }
        // For lower rows r >= d: (T * Mnd)[r, :] = Mnd_re[r, :]
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
        // Invert top d x d block of TM
        dense_inverse(d, TM_top.data());

        // Allocate and compute \hat{M}_{nd}
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

    // Return completed dimensionality reduction model
    return model;
}

/**
 * @brief Computes reduced trajectory state X(t) = exp(Q * t) * X(0) at continuous time t.
 */
std::vector<double> ReductionModel::solve_reduced(double t) const {
    // Form scaled matrix Q * t
    std::vector<double> Qt(static_cast<size_t>(d) * d);
    for (size_t i = 0; i < Qt.size(); ++i) {
        Qt[i] = Q[i] * t;
    }
    // Compute matrix exponential exp(Q * t)
    std::vector<double> expQt(static_cast<size_t>(d) * d);
    matrix_exp(d, Qt.data(), expQt.data());

    // Compute state vector X(t) = exp(Q * t) * X(0)
    std::vector<double> Xt(d);
    small_gemv(d, expQt.data(), X0.data(), Xt.data());
    // Return solved reduced vector
    return Xt;
}

/**
 * @brief Reconstructs full approximate probability distribution \tilde{P}(t) from reduced state X(t).
 */
std::vector<double> ReductionModel::reconstruct_solution(const std::vector<double>& X_t) const {
    // Validate that input vector size matches reduced dimension d
    if (static_cast<int>(X_t.size()) != d) {
        throw std::invalid_argument("X_t dimension does not match d");
    }

    // Allocate reconstructed distribution vector in re-indexed ordering
    std::vector<double> P_re(n, 0.0);

    if (transform == ReductionTransform::None) {
        // Direct reconstruction without transformation: \tilde{P}_re = Mnd_re * X(t)
        for (int64_t r = 0; r < n; ++r) {
            double s = 0.0;
            for (int c = 0; c < d; ++c) {
                s += Mnd_re[r * d + c] * X_t[c];
            }
            P_re[r] = s;
        }
    } else {
        // Left eigenbasis reconstruction: \tilde{P}_re = T^-1 * \hat{M}_re * X(t)
        // Step A: Intermediate vector Y = \hat{M}_re * X(t)
        std::vector<double> Y(n, 0.0);
        for (int64_t r = 0; r < n; ++r) {
            double s = 0.0;
            for (int c = 0; c < d; ++c) {
                s += Mhat_re[r * d + c] * X_t[c];
            }
            Y[r] = s;
        }
        // Step B: Apply inverse transformation T^-1 = [I_d, -Tnd_re[d:, :]^T; 0, I_m]
        // For dependent rows r < d: P_re[r] = Y[r] - sum_{m=d}^{n-1} Tnd_re[m, r] * Y[m]
        for (int r = 0; r < d; ++r) {
            double s = Y[r];
            for (int64_t m = d; m < n; ++m) {
                s -= Tnd_re[m * d + r] * Y[m];
            }
            P_re[r] = s;
        }
        // For independent rows r >= d: P_re[r] = Y[r]
        for (int64_t r = d; r < n; ++r) {
            P_re[r] = Y[r];
        }
    }

    // Invert permutation to map probabilities back to original state indices: P[perm[i]] = P_re[i]
    std::vector<double> P_orig(n);
    for (int64_t i = 0; i < n; ++i) {
        P_orig[perm[i]] = P_re[i];
    }
    // Return reconstructed probability distribution
    return P_orig;
}

/**
 * @brief Directly evaluates approximate probability distribution \tilde{P}(t) at time t.
 */
std::vector<double> ReductionModel::evaluate(double t) const {
    // Step 1: Solve reduced linear system X(t) = exp(Q * t) * X(0)
    std::vector<double> Xt = solve_reduced(t);
    // Step 2: Reconstruct full distribution \tilde{P}(t)
    return reconstruct_solution(Xt);
}

}  // namespace cme
