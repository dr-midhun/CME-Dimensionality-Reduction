#include "cme/alg6_7_8_fsp.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <stdexcept>

#include "cme/alg4_reduction.hpp"

namespace cme {

namespace {

// Combinatorial n_choose_k
double n_choose_k(int n, int k) {
    if (k < 0 || k > n) return 0.0;
    if (k == 0 || k == n) return 1.0;
    if (k == 1) return static_cast<double>(n);
    if (k == 2) return 0.5 * n * (n - 1);
    double res = 1.0;
    for (int i = 1; i <= k; ++i) res = res * (n - i + 1) / i;
    return std::round(res);
}

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

// Builds the truncated transition matrix D_JJ for a given state set NJ.
Csr build_truncated_D(const Network& net, const std::vector<std::vector<int>>& NJ,
                      const std::map<std::vector<int>, int32_t>& state_map) {
    const int32_t n = static_cast<int32_t>(NJ.size());
    const int h = net.m();

    std::vector<std::map<int32_t, double>> row_entries(n);
    for (int32_t i = 0; i < n; ++i) {
        row_entries[i][i] = 0.0;
    }

    std::vector<double> total_outflow(n, 0.0);

    for (int32_t j = 0; j < n; ++j) {
        const auto& state = NJ[j];
        for (int r = 0; r < h; ++r) {
            double prop = reaction_propensity(net, r, state);
            if (prop <= 0.0) continue;
            total_outflow[j] += prop;

            // Target state
            std::vector<int> target(net.k());
            for (int s = 0; s < net.k(); ++s) {
                target[s] = state[s] + net.S[r][s];
            }
            auto it = state_map.find(target);
            if (it != state_map.end()) {
                int32_t i = it->second;
                if (i != j) {
                    row_entries[i][j] += prop;
                }
            }
        }
    }

    for (int32_t j = 0; j < n; ++j) {
        row_entries[j][j] = -total_outflow[j];
    }

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

// Expands state space NJ by one reaction step in all reachable directions
void expand_state_space(const Network& net, std::vector<std::vector<int>>& NJ,
                        std::map<std::vector<int>, int32_t>& state_map) {
    const size_t current_size = NJ.size();
    const int h = net.m();
    for (size_t idx = 0; idx < current_size; ++idx) {
        std::vector<int> state = NJ[idx]; // copy to avoid reference invalidation on push_back
        for (int r = 0; r < h; ++r) {
            if (reaction_propensity(net, r, state) <= 0.0) continue;
            std::vector<int> target(net.k());
            for (int s = 0; s < net.k(); ++s) {
                target[s] = state[s] + net.S[r][s];
            }
            if (state_map.find(target) == state_map.end()) {
                int32_t new_idx = static_cast<int32_t>(NJ.size());
                state_map[target] = new_idx;
                NJ.push_back(target);
            }
        }
    }
}

// Uniformized / Krylov / Taylor series time stepper: P(t) = exp(D * dt) * P0
std::vector<double> propagate_distribution(const Csr& D, const std::vector<double>& P0, double dt) {
    const int n = D.n;
    if (n <= 0) return {};

    // Standard uniformization (Jensen's method) for continuous-time Markov chains:
    // With rate q >= max |D_ii|, let P_trans = I + D / q.
    // exp(D * dt) * P0 = sum_{k=0}^K exp(-q * dt) * (q * dt)^k / k! * (P_trans^k * P0)
    double q = 0.0;
    for (int r = 0; r < n; ++r) {
        for (int32_t p = D.row_ptr[r]; p < D.row_ptr[r + 1]; ++p) {
            if (D.col[p] == r) q = std::max(q, std::fabs(D.val[p]));
        }
    }
    if (q <= 0.0) return P0;

    double lambda = q * dt;
    // Poisson truncation threshold
    int k_max = static_cast<int>(lambda + 6.0 * std::sqrt(std::max(1.0, lambda)) + 20.0);
    if (k_max < 10) k_max = 10;

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

    // Compute Poisson probabilities poisson(k) = exp(-lambda) * lambda^k / k!
    std::vector<double> prob_poiss(k_max + 1, 0.0);
    double log_l = std::log(lambda);
    for (int k = 0; k <= k_max; ++k) {
        double log_p = -lambda + k * log_l - std::lgamma(k + 1.0);
        prob_poiss[k] = std::exp(log_p);
    }

    std::vector<double> result(n, 0.0);
    for (int i = 0; i < n; ++i) result[i] += prob_poiss[0] * current[i];

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

// Thesis Algorithm 6: Finite State Projection (Munsky & Khammash 2006)
FspResult alg6_fsp(const Network& net, const FspOptions& opt) {
    FspResult res;
    res.tf = opt.tf;

    // Initial truncated state space NJ: start with initial state net.x0
    std::vector<std::vector<int>> NJ = {net.x0};
    std::map<std::vector<int>, int32_t> state_map;
    state_map[net.x0] = 0;

    expand_state_space(net, NJ, state_map);

    bool flag = true;
    int expansions = 0;

    std::vector<double> P;

    while (flag && expansions < opt.max_expansions) {
        Csr DJ = build_truncated_D(net, NJ, state_map);

        std::vector<double> P0(NJ.size(), 0.0);
        P0[0] = 1.0; // mass concentrated at x0

        P = propagate_distribution(DJ, P0, opt.tf);

        double prob_sum = 0.0;
        for (double v : P) prob_sum += v;

        if (prob_sum >= 1.0 - opt.eps) {
            flag = false;
        } else {
            expand_state_space(net, NJ, state_map);
            ++expansions;
        }
    }

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

// Thesis Algorithm 7: Multiple Time-Step FSP with Fixed Time Step
FspResult alg7_multistep_fixed_fsp(const Network& net, const FspOptions& opt) {
    FspResult res;
    res.tf = opt.tf;
    const int n_int = (opt.n_intervals > 0) ? opt.n_intervals : 10;
    const double tau = opt.tf / n_int;
    const double eps_k = opt.eps / n_int;

    std::vector<std::vector<int>> NJ = {net.x0};
    std::map<std::vector<int>, int32_t> state_map;
    state_map[net.x0] = 0;
    expand_state_space(net, NJ, state_map);

    std::vector<double> P(NJ.size(), 0.0);
    P[0] = 1.0;

    int total_expansions = 0;

    for (int k = 0; k < n_int; ++k) {
        double current_eps = eps_k;
        bool step_ok = false;
        std::vector<double> P_next;

        while (!step_ok && total_expansions < opt.max_expansions) {
            Csr DJ = build_truncated_D(net, NJ, state_map);
            P_next = propagate_distribution(DJ, P, tau);

            double sum_next = 0.0;
            for (double v : P_next) sum_next += v;

            double sum_prev = 0.0;
            for (double v : P) sum_prev += v;

            // Loss in this interval: sum_prev - sum_next <= current_eps
            if ((sum_prev - sum_next) <= current_eps + 1e-14) {
                step_ok = true;
            } else {
                expand_state_space(net, NJ, state_map);
                ++total_expansions;
                P.resize(NJ.size(), 0.0);
            }
        }
        P = P_next;
    }

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

// Thesis Algorithm 8: Multiple Time-Step FSP with Variable Time Step
FspResult alg8_multistep_variable_fsp(const Network& net, const FspOptions& opt) {
    FspResult res;
    res.tf = opt.tf;

    std::vector<std::vector<int>> NJ = {net.x0};
    std::map<std::vector<int>, int32_t> state_map;
    state_map[net.x0] = 0;
    expand_state_space(net, NJ, state_map);

    std::vector<double> P(NJ.size(), 0.0);
    P[0] = 1.0;

    double tk = 0.0;
    int steps = 0;
    int total_expansions = 0;

    while (tk < opt.tf && steps < 1000) {
        double rem = opt.tf - tk;
        double eps_k = opt.eps * (tk + rem) / opt.tf; // cumulative error budget target
        double tau = rem;
        Csr DJ = build_truncated_D(net, NJ, state_map);
        std::vector<double> P_trial;

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
            tau *= 0.5;
        }

        tk += tau;
        P = P_trial;
        ++steps;
    }

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
