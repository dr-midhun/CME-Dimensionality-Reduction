#include "cme/alg6_7_8_fsp.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <stdexcept>

#include "cme/alg4_reduction.hpp"

namespace cme {

namespace {

/**
 * @brief Computes binomial coefficient (n choose k).
 *
 * @param n Available count of molecules.
 * @param k Stoichiometric coefficient required.
 * @return Binomial coefficient value.
 */
double n_choose_k(int n, int k) {
    if (k < 0 || k > n) return 0.0;
    if (k == 0 || k == n) return 1.0;
    if (k == 1) return static_cast<double>(n);
    if (k == 2) return 0.5 * n * (n - 1);
    double res = 1.0;
    for (int i = 1; i <= k; ++i) res = res * (n - i + 1) / i;
    return std::round(res);
}

/**
 * @brief Evaluates reaction propensity for reaction r given molecular count vector N.
 *
 * @param net Reaction network holding rates and stoichiometry.
 * @param r Zero-based index of reaction.
 * @param N State vector of molecular counts.
 * @return Transition propensity rate.
 */
double reaction_propensity(const Network& net, int r, const std::vector<int>& N) {
    double h = 1.0;
    for (int s = 0; s < net.k(); ++s) {
        int stoich = net.R[r][s];
        if (stoich > 0) {
            if (N[s] < stoich) return 0.0;
            h *= n_choose_k(N[s], stoich);
        }
    }
    return net.rate[r] * h;
}

/**
 * @brief Constructs the truncated transition rate matrix D_J over the subset state space N_J.
 *
 * Transitions leading outside N_J are treated as outflows (absorbing boundary loss),
 * which guarantees that the column sums satisfy \sum_i (D_J)_{ij} \le 0.
 *
 * @param net Reaction network.
 * @param NJ Current truncated collection of states.
 * @param state_map Map from state coordinate vector to index in NJ.
 * @return Truncated generator D_J in CSR format.
 */
Csr build_truncated_D(const Network& net, const std::vector<std::vector<int>>& NJ,
                      const std::map<std::vector<int>, int32_t>& state_map) {
    // Dimension of truncated state space
    const int32_t n = static_cast<int32_t>(NJ.size());
    // Total number of reaction channels
    const int h = net.m();

    // Sparse row accumulator: row_entries[i][j] = rate from j to i
    std::vector<std::map<int32_t, double>> row_entries(n);
    // Explicit diagonal entry for every row
    for (int32_t i = 0; i < n; ++i) {
        row_entries[i][i] = 0.0;
    }

    // Total outflow rate from each state j
    std::vector<double> total_outflow(n, 0.0);

    // Loop over each source state j in NJ
    for (int32_t j = 0; j < n; ++j) {
        const auto& state = NJ[j];
        // Evaluate all reaction channels
        for (int r = 0; r < h; ++r) {
            double prop = reaction_propensity(net, r, state);
            if (prop <= 0.0) continue;
            // Accumulate total outflow from state j
            total_outflow[j] += prop;

            // Form target state vector: target = state + S_r
            std::vector<int> target(net.k());
            for (int s = 0; s < net.k(); ++s) {
                target[s] = state[s] + net.S[r][s];
            }

            // Check if target state belongs to truncated set NJ
            auto it = state_map.find(target);
            if (it != state_map.end()) {
                int32_t i = it->second;
                // If internal transition, record propensity into row i column j
                if (i != j) {
                    row_entries[i][j] += prop;
                }
            }
            // If target state is outside NJ, probability leaks out (FSP absorbing boundary)
        }
    }

    // Set diagonal entries to negative total outflow: D_jj = - total_outflow[j]
    for (int32_t j = 0; j < n; ++j) {
        row_entries[j][j] = -total_outflow[j];
    }

    // Assemble CSR sparse matrix
    Csr D;
    D.n = n;
    D.row_ptr.assign(n + 1, 0);
    for (int32_t r = 0; r < n; ++r) {
        D.row_ptr[r + 1] = D.row_ptr[r] + static_cast<int32_t>(row_entries[r].size());
        for (const auto& kv : row_entries[r]) {
            D.col.push_back(kv.first);
            D.val.push_back(kv.second);
        }
    }
    return D;
}

/**
 * @brief Expands truncated state space N_J by 1 reaction step along all reachable directions.
 *
 * @param net Reaction network.
 * @param NJ Reference to state vector collection, expanded in place.
 * @param state_map Reference to state lookup map, updated in place.
 */
void expand_state_space(const Network& net, std::vector<std::vector<int>>& NJ,
                        std::map<std::vector<int>, int32_t>& state_map) {
    // Record current boundary size before adding new elements
    const size_t current_size = NJ.size();
    const int h = net.m();

    // Iterate over existing states
    for (size_t idx = 0; idx < current_size; ++idx) {
        // Take copy of state to avoid pointer invalidation when NJ reallocates
        std::vector<int> state = NJ[idx];
        for (int r = 0; r < h; ++r) {
            // Check if reaction can fire
            if (reaction_propensity(net, r, state) <= 0.0) continue;

            // Form reachable neighboring state
            std::vector<int> target(net.k());
            for (int s = 0; s < net.k(); ++s) {
                target[s] = state[s] + net.S[r][s];
            }

            // If neighbor is not yet in state space, insert it
            if (state_map.find(target) == state_map.end()) {
                int32_t new_idx = static_cast<int32_t>(NJ.size());
                state_map[target] = new_idx;
                NJ.push_back(target);
            }
        }
    }
}

/**
 * @brief Integrates truncated CME dP/dt = D_J * P using Jensen's Uniformization method.
 *
 * Uniformization avoids numerical stiffness and matrix inversion by expanding exp(D_J * dt)
 * into a Poisson-weighted sum of discrete transition matrix powers:
 *   exp(D_J * dt) * P0 = \sum_{k=0}^{K} e^{-\lambda} \frac{\lambda^k}{k!} (I + D_J / q)^k P0.
 *
 * @param D Truncated transition matrix in CSR format.
 * @param P0 Initial probability distribution vector.
 * @param dt Time step duration.
 * @return Propagated probability distribution vector at time dt.
 */
std::vector<double> propagate_distribution(const Csr& D, const std::vector<double>& P0, double dt) {
    const int n = D.n;
    if (n <= 0) return {};

    // Determine uniformization rate q >= max_i |D_{ii}|
    double q = 0.0;
    for (int r = 0; r < n; ++r) {
        for (int32_t p = D.row_ptr[r]; p < D.row_ptr[r + 1]; ++p) {
            if (D.col[p] == r) q = std::max(q, std::fabs(D.val[p]));
        }
    }
    // If matrix has zero outflow, distribution remains unchanged
    if (q <= 0.0) return P0;

    // Poisson mean parameter lambda = q * dt
    double lambda = q * dt;
    // Truncation limit K for Poisson summation: lambda + 6 * sqrt(lambda) + 20
    int k_max = static_cast<int>(lambda + 6.0 * std::sqrt(std::max(1.0, lambda)) + 20.0);
    if (k_max < 10) k_max = 10;

    // Sparse matrix-vector multiplier for uniformized transition operator P_trans = I + D / q
    auto spmv_trans = [&](const std::vector<double>& x, std::vector<double>& y) {
        y.assign(n, 0.0);
        for (int r = 0; r < n; ++r) {
            double s = 0.0;
            for (int32_t p = D.row_ptr[r]; p < D.row_ptr[r + 1]; ++p) {
                double v = D.val[p] / q;
                if (D.col[p] == r) v += 1.0;
                s += v * x[D.col[p]];
            }
            y[r] = s;
        }
    };

    std::vector<double> current = P0;
    std::vector<double> next(n);

    // Compute Poisson probabilities: prob_poiss[k] = exp(-lambda) * lambda^k / k!
    std::vector<double> prob_poiss(k_max + 1, 0.0);
    double log_l = std::log(lambda);
    for (int k = 0; k <= k_max; ++k) {
        double log_p = -lambda + k * log_l - std::lgamma(k + 1.0);
        prob_poiss[k] = std::exp(log_p);
    }

    // Accumulate k=0 term: prob_poiss[0] * P0
    std::vector<double> result(n, 0.0);
    for (int i = 0; i < n; ++i) result[i] += prob_poiss[0] * current[i];

    // Iteratively multiply by P_trans and accumulate Poisson weighted terms
    for (int k = 1; k <= k_max; ++k) {
        spmv_trans(current, next);
        current = next;
        double w = prob_poiss[k];
        for (int i = 0; i < n; ++i) {
            result[i] += w * current[i];
        }
    }

    return result;
}

}  // namespace

/**
 * @brief Implements Thesis Algorithm 6: Finite State Projection (Munsky & Khammash 2006).
 */
FspResult alg6_fsp(const Network& net, const FspOptions& opt) {
    FspResult res;
    res.tf = opt.tf;

    // Initialize truncated state space with initial seed state net.x0
    std::vector<std::vector<int>> NJ = {net.x0};
    std::map<std::vector<int>, int32_t> state_map;
    state_map[net.x0] = 0;

    // Perform initial state expansion to include direct neighbors
    expand_state_space(net, NJ, state_map);

    bool flag = true;
    int expansions = 0;
    std::vector<double> P;

    // Main FSP expansion loop: expand state space until probability loss <= eps
    while (flag && expansions < opt.max_expansions) {
        // Construct truncated infinitesimal generator D_J
        Csr DJ = build_truncated_D(net, NJ, state_map);

        // Place initial probability mass at starting state x0
        std::vector<double> P0(NJ.size(), 0.0);
        P0[0] = 1.0;

        // Propagate truncated CME distribution forward to tf
        P = propagate_distribution(DJ, P0, opt.tf);

        // Compute total preserved probability mass \sum P_J(tf)
        double prob_sum = 0.0;
        for (double v : P) prob_sum += v;

        // Check FSP convergence criterion: sum(P_J) >= 1 - eps
        if (prob_sum >= 1.0 - opt.eps) {
            flag = false;
        } else {
            // Expand truncated state space along reaction boundaries
            expand_state_space(net, NJ, state_map);
            ++expansions;
        }
    }

    // Assemble results
    res.states = NJ;
    res.prob = P;
    double sum = 0.0;
    for (double v : P) sum += v;
    res.error_bound = 1.0 - sum;
    res.expansions = expansions;
    res.steps = 1;
    res.converged = !flag;
    return res;
}

/**
 * @brief Implements Thesis Algorithm 7: Multiple Time-Step FSP with Fixed Time Step (Munsky & Khammash 2007).
 */
FspResult alg7_multistep_fixed_fsp(const Network& net, const FspOptions& opt) {
    FspResult res;
    res.tf = opt.tf;
    // Number of equal subintervals
    const int n_int = (opt.n_intervals > 0) ? opt.n_intervals : 10;
    // Step size tau = tf / n
    const double tau = opt.tf / n_int;
    // Error budget per interval: eps_k = eps / n
    const double eps_k = opt.eps / n_int;

    // Seed state space with initial state
    std::vector<std::vector<int>> NJ = {net.x0};
    std::map<std::vector<int>, int32_t> state_map;
    state_map[net.x0] = 0;
    expand_state_space(net, NJ, state_map);

    // Initial probability mass
    std::vector<double> P(NJ.size(), 0.0);
    P[0] = 1.0;

    int total_expansions = 0;

    // Loop through each subinterval k = 0 .. n_int - 1
    for (int k = 0; k < n_int; ++k) {
        double current_eps = eps_k;
        bool step_ok = false;
        std::vector<double> P_next;

        // Expand state space if probability loss during step tau exceeds error budget
        while (!step_ok && total_expansions < opt.max_expansions) {
            Csr DJ = build_truncated_D(net, NJ, state_map);
            P_next = propagate_distribution(DJ, P, tau);

            double sum_next = 0.0;
            for (double v : P_next) sum_next += v;

            double sum_prev = 0.0;
            for (double v : P) sum_prev += v;

            // Check if probability leakage during this interval is within budget: sum_prev - sum_next <= eps_k
            if ((sum_prev - sum_next) <= current_eps + 1e-14) {
                step_ok = true;
            } else {
                expand_state_space(net, NJ, state_map);
                ++total_expansions;
                P.resize(NJ.size(), 0.0);
            }
        }
        // Advance distribution to next step
        P = P_next;
    }

    // Assemble results
    res.states = NJ;
    res.prob = P;
    double sum = 0.0;
    for (double v : P) sum += v;
    res.error_bound = 1.0 - sum;
    res.expansions = total_expansions;
    res.steps = n_int;
    res.converged = (res.error_bound <= opt.eps * 1.01);
    return res;
}

/**
 * @brief Implements Thesis Algorithm 8: Multiple Time-Step FSP with Variable Time Step (Munsky & Khammash 2006).
 */
FspResult alg8_multistep_variable_fsp(const Network& net, const FspOptions& opt) {
    FspResult res;
    res.tf = opt.tf;

    // Seed state space
    std::vector<std::vector<int>> NJ = {net.x0};
    std::map<std::vector<int>, int32_t> state_map;
    state_map[net.x0] = 0;
    expand_state_space(net, NJ, state_map);

    std::vector<double> P(NJ.size(), 0.0);
    P[0] = 1.0;

    double tk = 0.0;
    int steps = 0;
    int total_expansions = 0;

    // Continue until time reaches tf
    while (tk < opt.tf && steps < 1000) {
        // Remaining simulation duration
        double rem = opt.tf - tk;
        // Cumulative allowable error budget target at tk + rem
        double eps_k = opt.eps * (tk + rem) / opt.tf;
        double tau = rem;
        Csr DJ = build_truncated_D(net, NJ, state_map);
        std::vector<double> P_trial;

        // Adaptive step loop: bisect tau until error bound holds
        while (tau > 1e-8) {
            P_trial = propagate_distribution(DJ, P, tau);
            double sum_trial = 0.0;
            for (double v : P_trial) sum_trial += v;

            // Algorithm 8 condition: 1 - ||P(tk + tau)||_1 <= eps_k
            if ((1.0 - sum_trial) <= eps_k + 1e-12) {
                break;
            }
            // If tau is already small, expand state space to reduce outflow
            if (tau < rem * 0.5 && total_expansions < opt.max_expansions) {
                expand_state_space(net, NJ, state_map);
                ++total_expansions;
                P.resize(NJ.size(), 0.0);
                DJ = build_truncated_D(net, NJ, state_map);
            }
            // Halve time step
            tau *= 0.5;
        }

        // Advance simulation time and update probability distribution
        tk += tau;
        P = P_trial;
        ++steps;
    }

    // Assemble results
    res.states = NJ;
    res.prob = P;
    double sum = 0.0;
    for (double v : P) sum += v;
    res.error_bound = 1.0 - sum;
    res.expansions = total_expansions;
    res.steps = steps;
    res.converged = (tk >= opt.tf - 1e-7);
    return res;
}

}  // namespace cme
