#pragma once
#include <cstdint>
#include <vector>
#include "cme/csr.hpp"

namespace cme {

// Transpose a CSR matrix.
// Input: n x n matrix D in CSR format (rows have ascending column indices).
// Output: n x n transposed matrix D^T in CSR format (rows also sorted by col).
Csr transpose_csr(const Csr& A);

}  // namespace cme
