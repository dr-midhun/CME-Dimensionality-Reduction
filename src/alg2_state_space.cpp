// Algorithm 2 of the thesis: all non-negative integer solutions of C N = b.
//
// Species are ordered by decreasing upper bound U, C is brought to RREF, the
// species at pivot columns ("dependent") are obtained from the non-pivot
// ("independent") ones, which are enumerated by depth-first search with the
// pruning rule  C_nonpivot * N_nonpivot <= b.  The thesis builds the
// candidate matrix explicitly; the DFS visits the same candidates without
// materialising them, and the first level is split across OpenMP threads.

#include "cme/state_space.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace cme {

/**
 * @brief Performs binary search on the sorted key vector to find state index.
 *
 * @param key The 64-bit hashed state key to locate.
 * @return Zero-based row/column index in transition matrix if found; otherwise -1.
 */
int64_t StateSpace::find(uint64_t key) const {
    // std::lower_bound returns iterator to first element >= key in sorted array
    auto it = std::lower_bound(keys.begin(), keys.end(), key);
    // If element is not found or points past the end, key is not present
    if (it == keys.end() || *it != key) return -1;
    // Compute zero-based distance from beginning of keys vector
    return it - keys.begin();
}

namespace {

/**
 * @brief Structure encapsulating the linear Diophantine constraint system C * N = b.
 */
struct Problem {
    int k = 0;                                 ///< Total number of chemical species
    int r0 = 0;                                ///< Number of conservation constraints (rows of C)
    std::vector<std::vector<int>> C;           ///< Conservation matrix rows
    std::vector<double> b;                     ///< Right-hand side vector: b = C * x0
    std::vector<int> U;                        ///< Coordinate upper bounds U_s
    std::vector<uint64_t> stride;              ///< Mixed-radix encoding strides
    std::vector<int> perm;                     ///< Column permutation: maps position in permuted C to original species index
    std::vector<std::vector<double>> A;        ///< Augmented matrix in Reduced Row Echelon Form [A_rref | b_rref]
    std::vector<int> pivot_pos;                ///< Column positions corresponding to pivot (dependent) variables
    std::vector<int> free_pos;                 ///< Column positions corresponding to free (independent) variables
};

/**
 * @brief Search context passed down the recursion stack in DFS enumeration.
 */
struct Ctx {
    const Problem* P = nullptr;                ///< Pointer to invariant problem definitions and matrices
    std::vector<int> N;                        ///< Current molecular counts assigned to free species
    std::vector<double> partial;               ///< Accumulated partial conservation product C_free * N_free
    std::vector<uint64_t>* out = nullptr;      ///< Thread-local output vector collecting valid state keys
};

/**
 * @brief Evaluates leaf node when all free variables have been assigned.
 *
 * Computes dependent (pivot) variables by back-substitution in RREF:
 *   N_{pivot} = b'_{pivot} - \sum_{j \in free} A'_{pivot, j} * N_j.
 * Verifies that all dependent values are non-negative integers and within bounds.
 *
 * @param c Search context containing current assignments.
 */
void leaf(Ctx& c) {
    const Problem& P = *c.P;
    // Allocate full species vector initialized to zero
    std::vector<int> full(P.k, 0);

    // Populate the assigned free species values into their original species indices
    for (int pos : P.free_pos) {
        full[P.perm[pos]] = c.N[P.perm[pos]];
    }

    // Number of pivot rows equals the matrix rank
    const int rank = static_cast<int>(P.pivot_pos.size());

    // Solve for each dependent variable in the RREF system
    for (int t = 0; t < rank; ++t) {
        // Start with the augmented right-hand side constant for row t
        double x = P.A[t][P.k];
        // Subtract contributions from all free variables
        for (int pos : P.free_pos) {
            x -= P.A[t][pos] * c.N[P.perm[pos]];
        }
        // Round to nearest integer to check for integrality
        double xr = std::nearbyint(x);
        // If solution is non-integer within numerical tolerance or negative, state is invalid
        if (std::fabs(x - xr) > 1e-6 || xr < 0) return;
        // Assign computed dependent variable to its original species index
        full[P.perm[P.pivot_pos[t]]] = static_cast<int>(xr);
    }

    // Validate that each coordinate respects its upper bound U_s and compute hash key
    uint64_t key = 0;
    for (int s = 0; s < P.k; ++s) {
        // Coordinate exceeds theoretical maximum bound
        if (full[s] > P.U[s]) return;
        // Accumulate mixed-radix key value: key += full[s] * stride[s]
        key += static_cast<uint64_t>(full[s]) * P.stride[s];
    }

    // Append verified valid state key to output collection
    c.out->push_back(key);
}

/**
 * @brief Depth-first search recursive routine to explore free variable combinations.
 *
 * Prunes branches early if partial conservation sum exceeds right-hand side b.
 *
 * @param c Current search context holding assignments and accumulator.
 * @param level Current recursion depth corresponding to index in P.free_pos.
 */
void dfs(Ctx& c, size_t level) {
    const Problem& P = *c.P;
    // If all free variables have been assigned, evaluate dependent variables at leaf
    if (level == P.free_pos.size()) {
        leaf(c);
        return;
    }

    // Retrieve original species index for the free variable at this depth
    int s = P.perm[P.free_pos[level]];
    // Save current partial product vector before exploring values
    std::vector<double> saved = c.partial;

    // Loop through all candidate integers for species s from 0 up to its upper bound U_s
    for (int v = 0; v <= P.U[s]; ++v) {
        bool ok = true;
        // Check conservation law constraints C * N <= b
        for (int i = 0; i < P.r0; ++i) {
            // Update partial sum with contribution of species s with value v
            c.partial[i] = saved[i] + static_cast<double>(P.C[i][s]) * v;
            // Since C is non-negative, if partial sum exceeds b[i], any higher v will also fail
            if (c.partial[i] > P.b[i] + 1e-9) {
                ok = false;
                break;
            }
        }
        // Prune search: stop exploring larger v values for this variable
        if (!ok) break;

        // Assign candidate value to free species s
        c.N[s] = v;
        // Recurse down to assign next free variable
        dfs(c, level + 1);
    }

    // Backtrack: restore saved partial sum and reset species assignment
    c.partial = saved;
    c.N[s] = 0;
}

}  // namespace

/**
 * @brief Implements Thesis Algorithm 2: Generates all reachable states satisfying conservation laws.
 *
 * Algorithm 2 pipeline:
 * 1. Compute RHS conservation vector b = C * x0.
 * 2. Calculate upper bound U_s for each species.
 * 3. Sort species descending by U_s (independent candidates first).
 * 4. Compute Reduced Row Echelon Form (RREF) to separate pivot and free variables.
 * 5. Run multithreaded DFS search over free variables with branch-and-bound pruning.
 * 6. Merge, sort, and deduplicate state keys into the final StateSpace structure.
 */
StateSpace generate_state_space(const Network& net) {
    Problem P;
    // Species count
    P.k = net.k();
    // Copy conservation matrix
    P.C = net.C;
    // Row count of conservation matrix
    P.r0 = static_cast<int>(net.C.size());
    // Initialize right-hand side vector b with zeros
    P.b.assign(P.r0, 0.0);

    // Compute RHS b_i = sum_s C_{is} * x0_s
    for (int i = 0; i < P.r0; ++i) {
        for (int s = 0; s < P.k; ++s) {
            P.b[i] += static_cast<double>(net.C[i][s]) * net.x0[s];
        }
    }

    // Compute theoretical upper bound for each species: U_s = floor(min_{i: C_is > 0} b_i / C_is)
    P.U.assign(P.k, -1);
    for (int s = 0; s < P.k; ++s) {
        double best = -1;
        for (int i = 0; i < P.r0; ++i) {
            if (P.C[i][s] > 0) {
                double bound = P.b[i] / P.C[i][s];
                if (best < 0 || bound < best) best = bound;
            }
        }
        // Fallback bound if unconstrained by conservation laws
        if (best < 0) best = 1000.0;
        P.U[s] = static_cast<int>(std::floor(best + 1e-9));
    }

    // Calculate mixed-radix strides: stride_0 = 1, stride_s = stride_{s-1} * (U_{s-1} + 1)
    P.stride.assign(P.k, 1);
    for (int s = 1; s < P.k; ++s) {
        P.stride[s] = P.stride[s - 1] * (static_cast<uint64_t>(P.U[s - 1]) + 1);
    }

    // Permute columns so species with larger upper bounds appear first
    P.perm.resize(P.k);
    std::iota(P.perm.begin(), P.perm.end(), 0);
    std::stable_sort(P.perm.begin(), P.perm.end(), [&](int a, int b) {
        return P.U[a] > P.U[b];
    });

    // Construct augmented matrix A of size r0 x (k + 1) using permuted species order
    P.A.assign(P.r0, std::vector<double>(P.k + 1, 0.0));
    for (int i = 0; i < P.r0; ++i) {
        for (int pos = 0; pos < P.k; ++pos) {
            P.A[i][pos] = P.C[i][P.perm[pos]];
        }
        P.A[i][P.k] = P.b[i];
    }

    // Perform Gaussian elimination to convert augmented matrix A to Reduced Row Echelon Form (RREF)
    int row = 0;
    std::vector<bool> is_pivot(P.k, false);

    for (int col = 0; col < P.k && row < P.r0; ++col) {
        // Find pivot element with maximum absolute value in current column
        int best_row = -1;
        double max_val = 1e-9;
        for (int r = row; r < P.r0; ++r) {
            if (std::fabs(P.A[r][col]) > max_val) {
                max_val = std::fabs(P.A[r][col]);
                best_row = r;
            }
        }
        // If column has no non-zero pivot, it is a free variable; skip to next column
        if (best_row < 0) continue;

        // Swap pivot row into current row position
        std::swap(P.A[row], P.A[best_row]);

        // Normalize pivot row so pivot element becomes 1.0
        double piv = P.A[row][col];
        for (int c = col; c <= P.k; ++c) P.A[row][c] /= piv;

        // Eliminate current column from all other rows
        for (int r = 0; r < P.r0; ++r) {
            if (r == row) continue;
            double factor = P.A[r][col];
            if (std::fabs(factor) > 1e-12) {
                for (int c = col; c <= P.k; ++c) {
                    P.A[r][c] -= factor * P.A[row][c];
                }
            }
        }

        // Record pivot position and mark column as pivot variable
        P.pivot_pos.push_back(col);
        is_pivot[col] = true;
        // Move to next row
        ++row;
    }

    // Non-pivot columns are designated as free (independent) variables
    for (int pos = 0; pos < P.k; ++pos) {
        if (!is_pivot[pos]) P.free_pos.push_back(pos);
    }

    // Prepare thread-local storage for parallel state enumeration
    std::vector<std::vector<uint64_t>> local_keys;

    #pragma omp parallel
    {
        // Each OpenMP thread allocates its private list of discovered state keys
        std::vector<uint64_t> thread_keys;
        Ctx ctx;
        ctx.P = &P;
        ctx.N.assign(P.k, 0);
        ctx.partial.assign(P.r0, 0.0);
        ctx.out = &thread_keys;

        if (P.free_pos.empty()) {
            // If zero free variables exist, only thread 0 evaluates the single unique leaf solution
            #pragma omp single
            {
                leaf(ctx);
            }
        } else {
            // First free variable is distributed across parallel threads
            int s0 = P.perm[P.free_pos[0]];
            #pragma omp for schedule(dynamic)
            for (int v = 0; v <= P.U[s0]; ++v) {
                // Initialize partial conservation product for first free variable
                std::vector<double> part(P.r0, 0.0);
                bool ok = true;
                for (int i = 0; i < P.r0; ++i) {
                    part[i] = static_cast<double>(P.C[i][s0]) * v;
                    if (part[i] > P.b[i] + 1e-9) { ok = false; break; }
                }
                if (!ok) continue;

                // Recursive DFS on remaining free variables starting from level 1
                ctx.partial = part;
                ctx.N[s0] = v;
                dfs(ctx, 1);
            }
        }

        // Safely collect thread-local state lists into master collection
        #pragma omp critical
        {
            local_keys.push_back(std::move(thread_keys));
        }
    }

    // Assemble final StateSpace result
    StateSpace ss;
    ss.k = P.k;
    ss.U = P.U;
    ss.stride = P.stride;

    // Calculate total state count across all threads to reserve memory upfront
    size_t total_keys = 0;
    for (const auto& lk : local_keys) total_keys += lk.size();
    ss.keys.reserve(total_keys);

    // Merge thread keys into single array
    for (auto& lk : local_keys) {
        ss.keys.insert(ss.keys.end(), lk.begin(), lk.end());
    }

    // Sort keys in ascending order to enable O(log N) binary search
    std::sort(ss.keys.begin(), ss.keys.end());
    // Remove potential duplicates
    ss.keys.erase(std::unique(ss.keys.begin(), ss.keys.end()), ss.keys.end());

    // Return complete enumerated state space
    return ss;
}

}  // namespace cme
