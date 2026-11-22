#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "cme/csr.hpp"
#include "cme/eigenbasis.hpp"

namespace cme {

// Transformation method for Algorithm 4:
// - None: T = I (Roussel & Zhu 2006, thesis Section 4.1 & 4.4.2)
// - LeftEigenbasis: T constructed from left eigenbasis Tnd (thesis Section 4.2.1, preserving probability conservation)
enum class ReductionTransform {
    None,            // T = I
    LeftEigenbasis   // T using left eigenbasis Tnd
};

struct ReductionOptions {
    int d = 2;                            // basis dimension
    ReductionTransform transform = ReductionTransform::LeftEigenbasis;
    double tol = 1e-10;                   // tolerance for eigenbasis iteration
    int max_iter = 200000;
    double cutoff = 0.0;                  // Chebyshev cutoff (if > 0)
    int degree = 100;
    Backend backend = Backend::Cpu;
    bool verbose = false;
};

struct ReductionModel {
    int64_t n = 0;
    int d = 0;
    ReductionTransform transform = ReductionTransform::LeftEigenbasis;

    // Reduced initial condition X0 (size d)
    std::vector<double> X0;

    // Reduced generator / transition matrix Qdd (d x d, row-major)
    std::vector<double> Q;

    // Permutation vector I (length n): maps re-indexed index -> original index
    std::vector<int64_t> perm;
    // Inverse permutation Irev (length n): maps original index -> re-indexed index
    std::vector<int64_t> perm_inv;

    // Right eigenbasis Mnd (n x d, row-major, in re-indexed space)
    std::vector<double> Mnd_re;
    // Left eigenbasis Tnd (n x d, row-major, in re-indexed space)
    std::vector<double> Tnd_re;

    // For LeftEigenbasis transform: Mhat = RREF(T * Mnd) (n x d, row-major, in re-indexed space)
    std::vector<double> Mhat_re;

    // Time evaluation functions
    // Computes reduced state X(t) = exp(Q * t) * X0
    std::vector<double> solve_reduced(double t) const;

    // Reconstructs approximate full probability distribution \tilde{P}(t) (length n, original indexing)
    std::vector<double> reconstruct_solution(const std::vector<double>& X_t) const;

    // Evaluates approximate solution \tilde{P}(t) at time t directly
    std::vector<double> solve(double t) const {
        return reconstruct_solution(solve_reduced(t));
    }
};

// Computes the reduced model using thesis Algorithm 4.
// Inputs:
//   D: CME transition matrix (n x n CSR)
//   P0: initial probability distribution (length n)
//   opt: reduction options (d, transform type, solver backend)
ReductionModel reduce_cme(const Csr& D, const std::vector<double>& P0, const ReductionOptions& opt);

// Utility: matrix exponential for small d x d row-major matrix using Padé approximation with scaling and squaring.
void matrix_exp(int d, const double* A, double* expA);

}  // namespace cme
