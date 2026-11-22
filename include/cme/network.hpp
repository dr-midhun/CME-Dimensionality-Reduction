#pragma once
#include <string>
#include <vector>

namespace cme {

// A mass-action reaction network plus the data Algorithm 2 needs.
//   S[i][s] : net stoichiometry of species s in reaction i   (thesis: S_i)
//   R[i][s] : reactant stoichiometry                         (thesis: R_i)
//   rate[i] : stochastic rate constant c_i
//   C       : non-negative conservation-law matrix (r x k), b = C * x0
struct Network {
    std::vector<std::string> species;
    std::vector<std::vector<int>> S, R;
    std::vector<double> rate;
    std::vector<std::vector<int>> C;
    std::vector<int> x0;
    int k() const { return static_cast<int>(species.size()); }
    int h() const { return static_cast<int>(rate.size()); }
    int m() const { return static_cast<int>(rate.size()); }
};

// Text format (see examples/michaelis_menten.net):
//   species S1 S2 S3 S4
//   reaction 1000 : S1 + S2 -> S3        # also: 2 S1 -> S2, A -> 0
//   reversible 1000 500 : S1 + S2 <-> S3
//   conserve 1 0 1 1                     # one row of C per line
//   initial 3 3 0 0
Network parse_network(const std::string& path);

}  // namespace cme
