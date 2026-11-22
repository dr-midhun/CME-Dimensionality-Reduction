#pragma once

#include <cmath>
#include <vector>

#include "cme/network.hpp"

namespace cme {

/**
 * @brief Computes the binomial coefficient (n choose k) for integer molecule counts.
 *
 * Evaluates stoichiometric reaction combinations according to Mass Action kinetics.
 *
 * @param n Available count of molecules.
 * @param k Required stoichiometric coefficient.
 * @return Value of n! / (k! * (n - k)!) as a double.
 */
inline double n_choose_k(int n, int k) {
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

/**
 * @brief Evaluates reaction propensity for reaction channel r given molecular count vector N.
 *
 * Follows standard stochastic chemical kinetics under the law of mass action.
 *
 * @param net Reaction network holding reaction rates and stoichiometric matrices.
 * @param r Zero-based index of reaction.
 * @param N State vector of molecular populations.
 * @return Transition propensity rate. Returns 0.0 if insufficient reactants.
 */
inline double reaction_propensity(const Network& net, int r, const std::vector<int>& N) {
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

}  // namespace cme
