#pragma once
#include <cstdint>
#include <vector>

namespace cme {

// Compressed sparse row matrix (32-bit indices; nnz must be < 2^31).
struct Csr {
    int32_t n = 0;
    std::vector<int32_t> row_ptr;  // size n+1
    std::vector<int32_t> col;      // size nnz, ascending within each row
    std::vector<double> val;       // size nnz
    int64_t nnz() const { return static_cast<int64_t>(col.size()); }
};

}  // namespace cme
