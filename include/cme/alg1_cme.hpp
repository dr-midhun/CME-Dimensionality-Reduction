#pragma once
#include <vector>
#include "cme/csr.hpp"
#include "cme/network.hpp"

namespace cme {

// Thesis Algorithm 1: Iterative breadth-first state space & transition matrix generation (Kan et al. 2016).
// Starting from an initial seed state x0, expands reachable states by applying reactions {Si},
// checks whether target states exist, assigns propensities, and constructs both the state space and matrix D.
struct Alg1Result {
    int k = 0;                          // number of species
    std::vector<std::vector<int>> states; // list of reachable states (size n x k)
    Csr D;                              // transition matrix (size n x n)
};

Alg1Result alg1_generate_cme(const Network& net);

}  // namespace cme
