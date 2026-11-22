// Algorithm 3 of the thesis: transition matrix of the CME.
//
// For every state N_v and reaction i: target N_v + S_i is located by key
// arithmetic + binary search (the thesis' index vector I), the propensity is
// c_i * prod_s nchoosek(N_s, R_is) (the thesis' C-hat), and the diagonal
// collects minus the total outflow.  Columns are assembled independently in
// parallel, then transposed to CSR (rows = target states).

#include "cme/transition.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace cme {
namespace {

/**
 * @brief Computes combinatorial factor (n choose r) for reaction stoichiometry.
 *
 * @param n Available molecular count of species.
 * @param r Reactant stoichiometry requirement.
 * @return Binomial coefficient n! / (r! * (n - r)!).
 */
inline double nchoosek(int n, int r) {
    // If requirement is 0, exactly 1 way
    if (r == 0) return 1.0;
    // If available count is less than required, 0 ways
    if (n < r) return 0.0;
    // Calculate product iteratively: prod_{j=0}^{r-1} (n - j) / (j + 1)
    double v = 1.0;
    for (int j = 0; j < r; ++j) {
        v = v * (n - j) / (j + 1);
    }
    return v;
}

}  // namespace

/**
 * @brief Generates transition rate matrix D in CSR format according to Thesis Algorithm 3.
 *
 * Parallelizes over source states v = 0 .. n-1. For each source state v:
 * 1. Decodes mixed-radix key into molecule count vector N.
 * 2. Loops through active reactions to evaluate transition propensities and target states.
 * 3. Uses precomputed key shifts to find target state indices via binary search in O(log n).
 * 4. Sums all outflows into diagonal entry D_{vv} = - \sum_{u \neq v} D_{uv}.
 * 5. Transposes the column-assembled entries into row-oriented Compressed Sparse Row (CSR) format.
 *
 * @param net Reaction network holding reaction rates, stoichiometric matrices S and R.
 * @param ss StateSpace holding reachable state keys and radix decoding parameters.
 * @return Csr Sparse matrix structure representing infinitesimal generator D.
 */
Csr build_transition_matrix(const Network& net, const StateSpace& ss) {
    // Number of species in the reaction system
    const int k = net.k();
    // Total number of discrete states in state space
    const int64_t n = ss.n();
    // Verify that state index fits inside a standard 32-bit signed integer
    if (n >= 2147483647ll) throw std::runtime_error("too many states for 32-bit indices");

    // Filter reactions that produce non-zero net stoichiometry change and precompute key shifts
    std::vector<int> rx;
    std::vector<int64_t> shift;

    for (int i = 0; i < net.h(); ++i) {
        bool null = true;
        int64_t sh = 0;
        for (int s = 0; s < k; ++s) {
            // Check if reaction i alters count of species s
            if (net.S[i][s] != 0) null = false;
            // Mixed-radix key change for reaction i: shift = sum_s S_{is} * stride_s
            sh += static_cast<int64_t>(net.S[i][s]) * static_cast<int64_t>(ss.stride[s]);
        }
        // Reactions with zero net change do not cause state transitions
        if (null) continue;
        rx.push_back(i);
        shift.push_back(sh);
    }

    // Number of reactions causing state transitions
    const int h = static_cast<int>(rx.size());
    // Maximum non-zero entries per column: h transitions + 1 diagonal entry
    const int W = h + 1;

    // Flat pre-allocated column buffers for thread-safe parallel generation
    std::vector<int32_t> trow(static_cast<size_t>(n) * W);
    std::vector<double> tval(static_cast<size_t>(n) * W);
    std::vector<int32_t> tcnt(n);

    #pragma omp parallel
    {
        // Thread-private state vector buffer
        std::vector<int> N(k);
        // Thread-private buffer for accumulating (target_row, propensity) pairs
        std::vector<std::pair<int32_t, double>> ent;

        // Statically distribute source states v across parallel threads
        #pragma omp for schedule(static)
        for (int64_t v = 0; v < n; ++v) {
            // Retrieve 64-bit mixed-radix hash key for source state v
            const uint64_t key = ss.keys[v];
            // Decode hash key into species molecular counts N
            ss.decode(key, N.data());
            // Clear entry buffer for state v
            ent.clear();
            // Initialize diagonal outflow accumulator
            double diag = 0.0;

            // Iterate over all active reaction channels
            for (int q = 0; q < h; ++q) {
                const int i = rx[q];
                // Base reaction rate constant c_i
                double prop = net.rate[i];

                // Compute combinatorial mass-action propensity: c_i * prod_s (N_s choose R_{is})
                for (int s = 0; s < k; ++s) {
                    if (net.R[i][s] > 0) prop *= nchoosek(N[s], net.R[i][s]);
                }
                // If propensity is zero, reaction cannot fire from state N
                if (prop == 0.0) continue;

                // Outflow accumulates every reaction propensity that can fire from state v
                diag -= prop;

                // Verify that target state coordinates stay within non-negative upper bounds
                bool inside = true;
                for (int s = 0; s < k; ++s) {
                    if (net.S[i][s] == 0) continue;
                    int t = N[s] + net.S[i][s];
                    if (t < 0 || t > ss.U[s]) { inside = false; break; }
                }
                if (!inside) continue;

                // Locate target state index u using precomputed key shift and binary search in O(log n)
                int64_t u = ss.find(static_cast<uint64_t>(static_cast<int64_t>(key) + shift[q]));
                // If target state is valid and within state space, record off-diagonal transition
                if (u >= 0) ent.push_back({static_cast<int32_t>(u), prop});
            }

            // Append diagonal entry: D_{vv} = - sum_{u != v} D_{uv}
            ent.push_back({static_cast<int32_t>(v), diag});

            // Sort entries for column v by target row index
            std::sort(ent.begin(), ent.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });

            // Merge entries with identical target row (reactions with identical net stoichiometric changes)
            int32_t cnt = 0;
            for (size_t e = 0; e < ent.size(); ++e) {
                if (cnt > 0 && trow[v * W + cnt - 1] == ent[e].first) {
                    // Accumulate propensity into existing row entry
                    tval[v * W + cnt - 1] += ent[e].second;
                } else {
                    // Store new target row and propensity
                    trow[v * W + cnt] = ent[e].first;
                    tval[v * W + cnt] = ent[e].second;
                    ++cnt;
                }
            }
            // Record number of non-zero entries in column v
            tcnt[v] = cnt;
        }
    }

    // Transpose column-oriented entries into CSR row-oriented format
    int64_t nnz = 0;
    // Count total non-zero entries across all columns
    for (int64_t v = 0; v < n; ++v) nnz += tcnt[v];

    // Allocate CSR matrix structure
    Csr D;
    D.n = static_cast<int32_t>(n);
    D.row_ptr.assign(n + 1, 0);
    D.col.resize(nnz);
    D.val.resize(nnz);

    // Count non-zeros per row in CSR format (row histogram)
    for (int64_t v = 0; v < n; ++v) {
        for (int32_t e = 0; e < tcnt[v]; ++e) {
            int32_t r = trow[v * W + e];
            D.row_ptr[r + 1]++;
        }
    }

    // Prefix sum to compute row start offsets in row_ptr
    for (int64_t r = 0; r < n; ++r) {
        D.row_ptr[r + 1] += D.row_ptr[r];
    }

    // Current insertion positions for each row in CSR value/column arrays
    std::vector<int32_t> head(D.row_ptr.begin(), D.row_ptr.begin() + n);

    // Populate CSR column indices and values by transposing column data
    for (int64_t v = 0; v < n; ++v) {
        for (int32_t e = 0; e < tcnt[v]; ++e) {
            int32_t r = trow[v * W + e];
            int32_t p = head[r]++;
            // In CSR format: column index is source state v, value is transition rate
            D.col[p] = static_cast<int32_t>(v);
            D.val[p] = tval[v * W + e];
        }
    }

    // Return final assembled infinitesimal generator transition matrix D
    return D;
}

}  // namespace cme
