#pragma once
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cme {

// In-place inverse of a small row-major d x d matrix (Gauss-Jordan, partial pivoting).
inline void dense_inverse(int d, double* A) {
    std::vector<double> inv(static_cast<size_t>(d) * d, 0.0);
    for (int i = 0; i < d; ++i) inv[i * d + i] = 1.0;
    for (int c = 0; c < d; ++c) {
        int p = c;
        for (int r = c + 1; r < d; ++r)
            if (std::fabs(A[r * d + c]) > std::fabs(A[p * d + c])) p = r;
        if (std::fabs(A[p * d + c]) < 1e-300)
            throw std::runtime_error("pivot block is singular (basis lost rank)");
        if (p != c)
            for (int j = 0; j < d; ++j) {
                std::swap(A[p * d + j], A[c * d + j]);
                std::swap(inv[p * d + j], inv[c * d + j]);
            }
        double piv = 1.0 / A[c * d + c];
        for (int j = 0; j < d; ++j) { A[c * d + j] *= piv; inv[c * d + j] *= piv; }
        for (int r = 0; r < d; ++r) {
            if (r == c) continue;
            double f = A[r * d + c];
            if (f == 0.0) continue;
            for (int j = 0; j < d; ++j) {
                A[r * d + j] -= f * A[c * d + j];
                inv[r * d + j] -= f * inv[c * d + j];
            }
        }
    }
    for (size_t i = 0; i < inv.size(); ++i) A[i] = inv[i];
}

}  // namespace cme
