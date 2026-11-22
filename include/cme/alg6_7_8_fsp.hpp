#pragma once
#include <vector>
#include "cme/csr.hpp"
#include "cme/network.hpp"

namespace cme {

/**
 * @brief Configuration parameters for Finite State Projection algorithms (Algorithms 6, 7, and 8).
 */
struct FspOptions {
    double tf = 1.0;            ///< Final simulation time (t_f > 0)
    double eps = 1e-3;          ///< Total allowable probability error tolerance \epsilon
    int n_intervals = 10;       ///< Number of time subintervals for fixed-step FSP (Algorithm 7)
    int max_expansions = 100;   ///< Safeguard ceiling on the number of state expansions
    bool verbose = false;       ///< Verbosity flag for diagnostic progress output
};

/**
 * @brief Output result structure returned by FSP algorithms (Algorithms 6, 7, and 8).
 */
struct FspResult {
    double tf = 0.0;                       ///< Final simulation time reached
    std::vector<std::vector<int>> states;  ///< Truncated state space N_J at termination
    std::vector<double> prob;              ///< Approximate probability distribution P_J(t_f)
    double error_bound = 0.0;              ///< Rigorous error upper bound: 1 - \sum P_J(t_f) \le \epsilon
    int expansions = 0;                    ///< Total number of state space expansion iterations performed
    int steps = 0;                         ///< Total number of time steps taken across the simulation
    bool converged = false;                ///< True if error bound was strictly satisfied within max_expansions
};

/**
 * @brief Thesis Algorithm 6: The Finite State Projection Algorithm (Munsky & Khammash 2006).
 *
 * Approximates the probability distribution at target time t_f on an iteratively expanded truncated state space N_J.
 * At each iteration:
 * 1. Constructs truncated transition matrix D_{J}.
 * 2. Integrates dP_J/dt = D_J * P_J from t=0 to t_f using uniformization.
 * 3. Checks if total preserved probability \sum P_J(t_f) \ge 1 - \epsilon.
 * 4. If test passes, returns P_J(t_f); if not, expands N_J to include all 1-step reachable neighboring states and repeats.
 *
 * @param net Reaction network specification.
 * @param opt FSP options (final time t_f, error tolerance eps).
 * @return FspResult containing final truncated state space and approximate distribution.
 */
FspResult alg6_fsp(const Network& net, const FspOptions& opt);

/**
 * @brief Thesis Algorithm 7: Multiple Time-Step FSP with Fixed Time Step \tau = t_f / n (Munsky & Khammash 2007).
 *
 * Divides time interval [0, t_f] into n equal subintervals of duration \tau = t_f / n.
 * Assigns an error budget \epsilon_k = \epsilon / n for each step.
 * Over each subinterval [t_k, t_k + \tau]:
 * 1. Integrates forward by step \tau.
 * 2. If probability loss exceeds \epsilon_k, expands state space N_J until error constraint is met.
 * 3. Propagates final distribution into initial condition for subsequent interval.
 *
 * @param net Reaction network specification.
 * @param opt FSP options (final time t_f, error tolerance eps, number of intervals).
 * @return FspResult holding final state space, probability distribution, and step counts.
 */
FspResult alg7_multistep_fixed_fsp(const Network& net, const FspOptions& opt);

/**
 * @brief Thesis Algorithm 8: Multiple Time-Step FSP with Variable Time Step (Munsky & Khammash 2006).
 *
 * Dynamically adjusts time step \tau based on error tolerance.
 * At each current time t_k:
 * 1. Sets initial candidate step \tau = t_f - t_k.
 * 2. Evaluates provisional distribution P(t_k + \tau).
 * 3. If error criterion 1 - \sum P(t_k + \tau) \le \epsilon_k holds, advances time t_k \leftarrow t_k + \tau.
 * 4. Otherwise, halves step \tau \leftarrow 0.5 \tau (and expands state space if \tau drops too low) and re-evaluates.
 *
 * @param net Reaction network specification.
 * @param opt FSP options (final time t_f, error tolerance eps).
 * @return FspResult holding final state space, distribution, and total adaptive steps.
 */
FspResult alg8_multistep_variable_fsp(const Network& net, const FspOptions& opt);

}  // namespace cme
