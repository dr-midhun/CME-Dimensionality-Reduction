#pragma once
#include <vector>
#include "cme/csr.hpp"
#include "cme/network.hpp"

namespace cme {

// Options for Finite State Projection algorithms (Algorithms 6, 7, 8).
struct FspOptions {
    double tf = 1.0;            // final simulation time
    double eps = 1e-3;          // allowable error tolerance
    int n_intervals = 10;       // number of intervals (for Algorithm 7 fixed-step)
    int max_expansions = 100;   // safeguard against unbounded state expansion
    bool verbose = false;
};

// Result returned by FSP algorithms
struct FspResult {
    double tf = 0.0;
    std::vector<std::vector<int>> states; // truncated state space NJ
    std::vector<double> prob;             // approximate probability distribution over NJ at tf
    double error_bound = 0.0;             // 1 - sum(prob)
    int expansions = 0;                   // number of state expansions performed
    int steps = 0;                        // number of time steps taken
    bool converged = false;
};

// Thesis Algorithm 6: The Finite State Projection Algorithm (Munsky & Khammash 2006).
// Approximates the probability distribution at time tf on a truncated state space NJ.
// If 1^T P_J(tf) < 1 - eps, expands the state space by reachable neighbors and repeats.
FspResult alg6_fsp(const Network& net, const FspOptions& opt);

// Thesis Algorithm 7: Multiple Time-Step FSP with Fixed Time Step tau = tf / n (Munsky & Khammash 2007).
// Steps across sub-intervals [t_k, t_k + tau], updating and expanding state space if error exceeds sub-budget eps_k.
FspResult alg7_multistep_fixed_fsp(const Network& net, const FspOptions& opt);

// Thesis Algorithm 8: Multiple Time-Step FSP with Variable Time Step (Munsky & Khammash 2006).
// Assesses remaining time tau = tf - tk; if 1^T P(tk + tau) < 1 - eps_k, halves tau until error tolerance holds.
FspResult alg8_multistep_variable_fsp(const Network& net, const FspOptions& opt);

}  // namespace cme
