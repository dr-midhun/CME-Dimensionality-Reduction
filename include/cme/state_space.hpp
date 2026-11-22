#pragma once
#include <cstdint>
#include <vector>
#include "cme/network.hpp"

namespace cme {

// State space of the CME (thesis Algorithm 2), stored compactly as sorted
// mixed-radix keys:  key(N) = sum_s N_s * stride_s,  0 <= N_s <= U_s.
// A state is decoded on the fly, so memory is 8 bytes per state and
// lookup of a neighbouring state is a binary search (GPU friendly).
struct StateSpace {
    int k = 0;
    std::vector<int> U;
    std::vector<uint64_t> stride;
    std::vector<uint64_t> keys;  // sorted ascending; index = row/column in D

    int64_t n() const { return static_cast<int64_t>(keys.size()); }

    void decode(uint64_t key, int* N) const {
        for (int s = 0; s < k; ++s) {
            uint64_t radix = static_cast<uint64_t>(U[s]) + 1;
            N[s] = static_cast<int>((key / stride[s]) % radix);
        }
    }
    int64_t find(uint64_t key) const;  // -1 if absent
};

// All non-negative integer solutions of C N = b (b = C x0).
StateSpace generate_state_space(const Network& net);

}  // namespace cme
