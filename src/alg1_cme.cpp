#include "cme/alg1_cme.hpp"
#include <cmath>
#include <map>
#include <stdexcept>

namespace cme {

namespace {

/**
 * @brief Computes the binomial coefficient (n choose k) for integer molecule counts.
 *
 * Used in evaluating stoichiometric reaction combinations according to Mass Action kinetics.
 *
 * @param n Total available molecules of a given reactant species.
 * @param k Required stoichiometric molecules of the species in the reaction.
 * @return Value of n! / (k! * (n - k)!) as a double.
 */
double n_choose_k(int n, int k) {
    // If fewer molecules are required than 0 or more than available n, combination is impossible (0)
    if (k < 0 || k > n) return 0.0;
    // Base cases: choosing 0 or all n items yields exactly 1 combination
    if (k == 0 || k == n) return 1.0;
    // Choosing 1 item yields n combinations
    if (k == 1) return static_cast<double>(n);
    // Choosing 2 items yields n*(n-1)/2
    if (k == 2) return 0.5 * n * (n - 1);
    // Choosing 3 items yields n*(n-1)*(n-2)/6
    if (k == 3) return (n * (n - 1.0) * (n - 2.0)) / 6.0;

    // General product formula: prod_{i=1}^k (n - i + 1) / i
    double res = 1.0;
    for (int i = 1; i <= k; ++i) {
        // Multiplies by the decreasing numerator factor and divides by the current factorial step
        res = res * (n - i + 1) / i;
    }
    // Round to nearest integer to eliminate floating point accumulation inaccuracies
    return std::round(res);
}

/**
 * @brief Evaluates the reaction propensity h_r(N) * c_r for reaction r given current state vector N.
 *
 * Implements standard stochastic chemical kinetics under the law of mass action.
 *
 * @param net Reaction network holding reaction rates and stoichiometric matrices.
 * @param r Zero-based index of the reaction to evaluate.
 * @param N State vector of molecular populations for all species.
 * @return Transition propensity (rate) for reaction r. Returns 0.0 if insufficient reactants.
 */
double reaction_propensity(const Network& net, int r, const std::vector<int>& N) {
    // Initialize combinatorial combination counter to 1.0
    double h = 1.0;

    // Loop through each species s in the network
    for (int s = 0; s < net.k(); ++s) {
        // Look up the required reactant stoichiometry for species s in reaction r
        int stoich = net.R[r][s];
        // If this species is a reactant in reaction r
        if (stoich > 0) {
            // If the current molecule count is strictly less than required stoichiometry, reaction cannot fire
            if (N[s] < stoich) return 0.0;
            // Multiply combination factor by (N[s] choose stoich)
            h *= n_choose_k(N[s], stoich);
        }
    }
    // Propensity equals the reaction rate constant multiplied by the combinatorial factor h
    return net.rate[r] * h;
}

}  // namespace

/**
 * @brief Generates state space and infinitesimal generator D using Thesis Algorithm 1.
 *
 * Follows BFS traversal from initial state x0, dynamically discovering new states
 * and establishing state-to-state transition rates until no new reachable states are found.
 */
Alg1Result alg1_generate_cme(const Network& net) {
    // Validate network: must contain at least one species
    if (net.species.empty()) throw std::invalid_argument("Empty network");

    // Cache number of chemical species (k) and number of reaction channels (h)
    const int k = net.k();
    const int h = net.m();

    // Prepare result object
    Alg1Result res;
    res.k = k;

    // Map each state vector to its integer index in res.states for O(log N) lookup
    std::map<std::vector<int>, int32_t> state_map;

    // Initialize state space with the starting state N1 = net.x0
    std::vector<int> N1 = net.x0;
    // Append initial state as index 0
    res.states.push_back(N1);
    // Record index 0 in the lookup map
    state_map[N1] = 0;

    // Data structure to hold raw transition quadruples / triples: (target_i, source_j, propensity)
    struct Trans {
        int32_t i; ///< Target state row index
        int32_t j; ///< Source state column index
        double val; ///< Propensity transition rate
    };
    // List of all discovered transitions between reachable states
    std::vector<Trans> transitions;

    // BFS queue index pointer j represents current source state being expanded
    int32_t j = 0;
    // Iterate while there are unexplored states in the queue
    while (j < static_cast<int32_t>(res.states.size())) {
        // Copy the current source state Nj
        std::vector<int> Nj = res.states[j];

        // Evaluate all reaction channels r from 0 to h-1
        for (int r = 0; r < h; ++r) {
            // Compute transition rate from state Nj under reaction r
            double prop = reaction_propensity(net, r, Nj);
            // If reaction has zero probability of firing from state Nj, skip
            if (prop <= 0.0) continue;

            // Formulate target state Nk = Nj + S_r (adding net stoichiometric change)
            std::vector<int> Nk(k);
            for (int s = 0; s < k; ++s) {
                // Apply net stoichiometric change for species s
                Nk[s] = Nj[s] + net.S[r][s];
            }

            // Check if target state Nk already exists in the state space
            int32_t i_target = -1;
            auto it = state_map.find(Nk);
            if (it != state_map.end()) {
                // State already discovered; retrieve existing index
                i_target = it->second;
            } else {
                // New state discovered; assign next available index
                i_target = static_cast<int32_t>(res.states.size());
                // Append newly discovered state to list of states
                res.states.push_back(Nk);
                // Insert new state and index into lookup map
                state_map[Nk] = i_target;
            }

            // Record directed transition from source state j to target state i_target with propensity
            transitions.push_back({i_target, j, prop});
        }
        // Advance queue pointer to next discovered state
        ++j;
    }

    // Total number of discovered reachable states
    const int32_t n = static_cast<int32_t>(res.states.size());

    // Allocate array to track total probability outflow from each source state j
    std::vector<double> outflow(n, 0.0);
    // Structure to group non-zero matrix elements by row i: row_entries[i][j] = rate
    std::vector<std::map<int32_t, double>> row_entries(n);

    // Initialize explicit diagonal entry (row i == col i) for every state
    for (int32_t i = 0; i < n; ++i) {
        row_entries[i][i] = 0.0;
    }

    // Accumulate off-diagonal transition propensities
    for (const auto& tr : transitions) {
        // Only process distinct source and target states for off-diagonal rates
        if (tr.i != tr.j) {
            // In master equation dP/dt = D P, D_ij is rate from state j into state i
            row_entries[tr.i][tr.j] += tr.val;
            // Outflow from state j increases by this transition rate
            outflow[tr.j] += tr.val;
        }
    }

    // Diagonal elements D_jj must equal negative sum of all outflows from state j
    for (int32_t j_idx = 0; j_idx < n; ++j_idx) {
        // Set diagonal element D_jj = - sum_{i != j} D_ij
        row_entries[j_idx][j_idx] = -outflow[j_idx];
    }

    // Set matrix dimension in CSR structure
    res.D.n = n;
    // Initialize row_ptr array of size n + 1 with zeros
    res.D.row_ptr.assign(n + 1, 0);

    // Assemble Compressed Sparse Row (CSR) arrays
    for (int32_t r = 0; r < n; ++r) {
        // row_ptr[r + 1] = row_ptr[r] + number of non-zero entries in row r
        res.D.row_ptr[r + 1] = res.D.row_ptr[r] + static_cast<int32_t>(row_entries[r].size());
        // Iterate through all columns in row r
        for (const auto& kv : row_entries[r]) {
            // Append column index
            res.D.col.push_back(kv.first);
            // Append matrix coefficient value
            res.D.val.push_back(kv.second);
        }
    }

    // Return completed state space and transition matrix result
    return res;
}

}  // namespace cme
