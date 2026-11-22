#pragma once
#include <vector>
#include "cme/csr.hpp"
#include "cme/network.hpp"

namespace cme {

/**
 * @brief Output result of Thesis Algorithm 1 (Iterative State Space & Transition Matrix Generation).
 *
 * Contains:
 * - k: total number of chemical species in the reaction network.
 * - states: list of reachable state vectors (dimension n x k), indexed 0 through n-1.
 * - D: infinitesimal generator / transition rate matrix in Compressed Sparse Row (CSR) format (dimension n x n).
 */
struct Alg1Result {
    int k = 0;                            ///< Number of chemical species
    std::vector<std::vector<int>> states; ///< Enumerated state vectors N_j
    Csr D;                                ///< Infinitesimal generator transition matrix in CSR format
};

/**
 * @brief Implements Thesis Algorithm 1: Iterative breadth-first state space & transition matrix generation.
 *
 * Reference: Thesis Chapter 2 / Algorithm 1 (Kan et al. 2016).
 * Starting from an initial seed state x0, this algorithm systematically explores all reachable states
 * using a breadth-first search (BFS) queue. For each discovered state, propensities for all reactions
 * are computed. Valid transitions generate target states and populate the off-diagonal elements of
 * matrix D. Diagonal elements are set to the negative sum of column outflows to preserve probability.
 *
 * @param net Reaction network specification containing stoichiometric matrix S, reactant matrix R,
 *            rate constants, and initial state x0.
 * @return Alg1Result containing the enumerated state list and the CSR transition matrix D.
 */
Alg1Result alg1_generate_cme(const Network& net);

}  // namespace cme
