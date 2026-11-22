#include "cme/csr_util.hpp"
#include <algorithm>

namespace cme {

Csr transpose_csr(const Csr& A) {
    Csr At;
    At.n = A.n;
    const int32_t n = A.n;
    At.row_ptr.assign(n + 1, 0);

    // Count entries per column of A (row of At)
    for (int32_t col_idx : A.col) {
        ++At.row_ptr[col_idx + 1];
    }
    for (int32_t i = 0; i < n; ++i) {
        At.row_ptr[i + 1] += At.row_ptr[i];
    }

    const int64_t nnz = A.nnz();
    At.col.resize(nnz);
    At.val.resize(nnz);

    std::vector<int32_t> next_pos = At.row_ptr;
    for (int32_t r = 0; r < n; ++r) {
        for (int32_t p = A.row_ptr[r]; p < A.row_ptr[r + 1]; ++p) {
            int32_t c = A.col[p];
            int32_t dest = next_pos[c]++;
            At.col[dest] = r;
            At.val[dest] = A.val[p];
        }
    }
    return At;
}

}  // namespace cme
