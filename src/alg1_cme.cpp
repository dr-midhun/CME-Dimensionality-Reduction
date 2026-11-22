#include "cme/alg1_cme.hpp"
#include <cmath>
#include <map>
#include <stdexcept>

namespace cme {

namespace {

// Combinatorial n_choose_k for integer propensities
double n_choose_k(int n, int k) {
    if (k < 0 || k > n) return 0.0;
    if (k == 0 || k == n) return 1.0;
    if (k == 1) return static_cast<double>(n);
    if (k == 2) return 0.5 * n * (n - 1);
    if (k == 3) return (n * (n - 1.0) * (n - 2.0)) / 6.0;
    double res = 1.0;
    for (int i = 1; i <= k; ++i) {
        res = res * (n - i + 1) / i;
    }
    return std::round(res);
}

// Compute propensity of reaction r from state N
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

}  // namespace

Alg1Result alg1_generate_cme(const Network& net) {
    if (net.species.empty()) throw std::invalid_argument("Empty network");
    const int k = net.k();
    const int h = net.m();

    Alg1Result res;
    res.k = k;

    // Map state -> index in states
    std::map<std::vector<int>, int32_t> state_map;

    // Seed with N1 = net.x0
    std::vector<int> N1 = net.x0;
    res.states.push_back(N1);
    state_map[N1] = 0;

    // Adjacency / transition records: (source_j, target_i, propensity)
    struct Trans {
        int32_t i; // target row
        int32_t j; // source col
        double val;
    };
    std::vector<Trans> transitions;

    int32_t j = 0;
    while (j < static_cast<int32_t>(res.states.size())) {
        std::vector<int> Nj = res.states[j];

        for (int r = 0; r < h; ++r) {
            double prop = reaction_propensity(net, r, Nj);
            if (prop <= 0.0) continue;

            // Target state Nk = Nj + Sr
            std::vector<int> Nk(k);
            for (int s = 0; s < k; ++s) {
                Nk[s] = Nj[s] + net.S[r][s];
            }

            int32_t i_target = -1;
            auto it = state_map.find(Nk);
            if (it != state_map.end()) {
                i_target = it->second;
            } else {
                i_target = static_cast<int32_t>(res.states.size());
                res.states.push_back(Nk);
                state_map[Nk] = i_target;
            }

            transitions.push_back({i_target, j, prop});
        }
        ++j;
    }

    const int32_t n = static_cast<int32_t>(res.states.size());

    // Build CSR transition matrix D (rows = target i, cols = source j)
    // Diagonal entry D_jj = - sum_{i != j} D_ij (total outflow from state j)
    std::vector<double> outflow(n, 0.0);
    // Group entries by row i: map col j -> sum of propensities
    std::vector<std::map<int32_t, double>> row_entries(n);

    // Explicit diagonal entry for every state (row i == col i)
    for (int32_t i = 0; i < n; ++i) {
        row_entries[i][i] = 0.0;
    }

    for (const auto& tr : transitions) {
        if (tr.i != tr.j) {
            row_entries[tr.i][tr.j] += tr.val;
            outflow[tr.j] += tr.val;
        }
    }

    for (int32_t j_idx = 0; j_idx < n; ++j_idx) {
        row_entries[j_idx][j_idx] = -outflow[j_idx];
    }

    res.D.n = n;
    res.D.row_ptr.assign(n + 1, 0);
    for (int32_t r = 0; r < n; ++r) {
        res.D.row_ptr[r + 1] = res.D.row_ptr[r] + static_cast<int32_t>(row_entries[r].size());
        for (const auto& kv : row_entries[r]) {
            res.D.col.push_back(kv.first);
            res.D.val.push_back(kv.second);
        }
    }

    return res;
}

}  // namespace cme
