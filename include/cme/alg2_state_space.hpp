#pragma once
#include <cstdint>
#include <vector>
#include "cme/network.hpp"

namespace cme {

/**
 * @brief Compact representation of the reaction network state space (Thesis Algorithm 2).
 *
 * Implements mixed-radix coordinate hashing:
 *   key(N) = \sum_{s=0}^{k-1} N_s \times stride_s, where 0 <= N_s <= U_s.
 * Each state is stored as a 64-bit integer key in a sorted list.
 * Membership test and neighbor indexing are performed in O(log n) time via binary search.
 */
struct StateSpace {
    int k = 0;                            ///< Number of chemical species
    std::vector<int> U;                   ///< Coordinate upper bounds U_s for each species s
    std::vector<uint64_t> stride;         ///< Mixed-radix strides for encoding: stride_s = \prod_{j=0}^{s-1} (U_j + 1)
    std::vector<uint64_t> keys;           ///< Sorted unique state keys; array index corresponds to state index (0 to n-1)

    /**
     * @brief Returns total number of valid reachable states in the state space.
     */
    int64_t n() const { return static_cast<int64_t>(keys.size()); }

    /**
     * @brief Decodes a 64-bit state key back into its multi-dimensional species count vector N.
     *
     * @param key 64-bit hash key of the state.
     * @param N Output buffer array of length k to receive molecular counts.
     */
    void decode(uint64_t key, int* N) const {
        for (int s = 0; s < k; ++s) {
            uint64_t radix = static_cast<uint64_t>(U[s]) + 1;
            N[s] = static_cast<int>((key / stride[s]) % radix);
        }
    }

    /**
     * @brief Encodes a multi-dimensional species count vector N into a 64-bit state key.
     *
     * @param N Input species count vector of length k.
     * @return 64-bit mixed-radix hash key.
     */
    uint64_t encode(const int* N) const {
        uint64_t key = 0;
        for (int s = 0; s < k; ++s) {
            key += static_cast<uint64_t>(N[s]) * stride[s];
        }
        return key;
    }

    /**
     * @brief Finds the zero-based index of a state given its 64-bit key using binary search.
     *
     * @param key State key to search for.
     * @return Index in keys array (0 <= index < n), or -1 if the state is not in the state space.
     */
    int64_t find(uint64_t key) const;
};

/**
 * @brief Generates all non-negative integer solutions of C * N = b = C * x0 (Thesis Algorithm 2).
 *
 * Implements RREF reduction, greedy upper-bound ordering, and depth-first search (DFS) with branch-and-bound pruning.
 *
 * @param net Reaction network holding conservation matrix C and initial state x0.
 * @return Complete StateSpace instance holding sorted state keys and decoding strides.
 */
StateSpace generate_state_space(const Network& net);

}  // namespace cme
