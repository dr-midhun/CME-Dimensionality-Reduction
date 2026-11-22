#include "cme/csr_util.hpp"
#include <algorithm>

namespace cme {

/**
 * @brief Computes the transpose of a square matrix stored in Compressed Sparse Row (CSR) format.
 *
 * Transposing a CSR matrix exchanges row and column indices. The algorithm uses a two-pass
 * counting sort to compute the row pointers of A^T, and then scatters the entries in O(nnz) time.
 *
 * @param A The input CSR matrix of dimension n x n.
 * @return Csr Transposed matrix A^T in CSR format.
 */
Csr transpose_csr(const Csr& A) {
    Csr At;
    // Set matrix dimension
    At.n = A.n;
    const int32_t n = A.n;
    // Allocate row_ptr of size n + 1 initialized to zeros
    At.row_ptr.assign(n + 1, 0);

    // Pass 1: Count entries per column of A (which become rows of At)
    for (int32_t col_idx : A.col) {
        // Increment non-zero count for column col_idx
        ++At.row_ptr[col_idx + 1];
    }
    // Prefix sum to compute row offsets for At
    for (int32_t i = 0; i < n; ++i) {
        At.row_ptr[i + 1] += At.row_ptr[i];
    }

    // Allocate memory for column indices and values
    const int64_t nnz = A.nnz();
    At.col.resize(nnz);
    At.val.resize(nnz);

    // Track write position for each row of At
    std::vector<int32_t> next_pos = At.row_ptr;
    // Pass 2: Scatter entries from A into At
    for (int32_t r = 0; r < n; ++r) {
        for (int32_t p = A.row_ptr[r]; p < A.row_ptr[r + 1]; ++p) {
            int32_t c = A.col[p];
            // Destination offset in At
            int32_t dest = next_pos[c]++;
            // Inverted indices: column of At is row r of A
            At.col[dest] = r;
            At.val[dest] = A.val[p];
        }
    }
    return At;
}

}  // namespace cme
