#pragma once
#include <vector>
#include "cme/csr.hpp"
#include "cme/network.hpp"

namespace cme {

/**
 * @brief Output result of Preconditioning Algorithm (Thesis Algorithm 10).
 *
 * Holds the re-indexed transition matrix, reduced generator Q, left semi-orthogonal eigenbasis T,
 * reduced initial condition Z(0), and the optimal continuous time step tau where truncation error Z_eps(tau) <= eps.
 */
struct PreconditionResult {
    Csr D;                                ///< Re-indexed transition matrix in CSR format
    int d = 0;                            ///< Reduced subspace dimension
    std::vector<int64_t> perm;            ///< Permutation vector I mapping re-indexed -> original states
    std::vector<double> Tnd_re;           ///< Left semi-orthogonal eigenbasis (size n x d, row-major)
    std::vector<double> Q;                ///< Reduced generator matrix Q_dd (size d x d, row-major)
    std::vector<double> Z0;               ///< Reduced initial condition Z(0) = Tnd^T * P0 (size d)
    double tau = 0.0;                     ///< Optimal time-step duration satisfying error bound
    std::vector<double> Z_tau;            ///< Evaluated reduced state vector Z(tau) = exp(Q * tau) * Z(0)
};

/**
 * @brief Output result of Approximation Algorithm (Thesis Algorithm 11).
 *
 * Reconstructs the approximate probability distribution P_{k+1} at the end of the time step.
 */
struct ApproximationResult {
    std::vector<double> P_next;           ///< Reconstructed approximate probability vector over current truncated states
    std::vector<double> Mnd_re;           ///< Right semi-orthogonal eigenbasis Mnd (size n x d, row-major)
};

/**
 * @brief Snapshot of subinterval parameters accumulated during multi-time-step integration (Algorithm 12).
 *
 * Records local state space, reduced operator Q, and basis matrices to allow continuous transient evaluation at any query time t.
 */
struct TransientStepSnapshot {
    double t_start = 0.0;                 ///< Start time of subinterval t_k
    double t_end = 0.0;                   ///< End time of subinterval t_k + tau_k
    double tau = 0.0;                     ///< Time step length tau_k
    int d = 0;                            ///< Reduced basis dimension
    std::vector<int64_t> perm;            ///< State permutation vector for this interval
    std::vector<std::vector<int>> states; ///< Truncated state space N_k active in this interval
    std::vector<double> Tnd_re;           ///< Left eigenbasis T_k in re-indexed space
    std::vector<double> Mnd_re;           ///< Right eigenbasis M_k in re-indexed space
    std::vector<double> Q;                ///< Reduced generator Q_k
    std::vector<double> Z0;               ///< Starting reduced state vector Z_k(0)
};

/**
 * @brief Output result of Preconditioned Transient FSP (Thesis Algorithm 9).
 */
struct PreconditionedFspResult {
    double tf = 0.0;                           ///< Final time reached
    std::vector<std::vector<int>> full_states; ///< Union of all states reached across all intervals (N_full)
    std::vector<double> P_tf;                  ///< Continuous probability distribution P(tf) over N_full
    std::vector<TransientStepSnapshot> steps;  ///< Ordered snapshot history of all subinterval steps
    int num_subintervals = 0;                  ///< Number of subintervals executed
    bool converged = false;                    ///< True if simulation successfully covered [t0, tf]
};

/**
 * @brief Thesis Algorithm 10: Preconditioning Algorithm.
 *
 * Reference: Thesis Section 5.1 & Algorithm 10.
 * 1. Computes left eigenbasis Tnd of D^T for slowest decaying modes.
 * 2. Selects d pivot rows and constructs permutation I.
 * 3. Normalizes Tnd so its top d x d block is Identity.
 * 4. Forms reduced transition generator Q = Tnd^T * D[:, 0:d].
 * 5. Sets up augmented error ODE system [ \dot{Z} ; \dot{Z_eps} ] = [ Q, 0 ; -1^T Q, 0 ] [ Z ; Z_eps ].
 * 6. Searches for largest valid time-step tau <= t_eps where error Z_eps(tau) <= eps.
 *
 * @param D Truncated transition rate matrix in CSR format (n x n).
 * @param P0 Starting probability distribution over truncated states (length n).
 * @param d Target reduced basis dimension.
 * @param eps Error threshold.
 * @param t_eps Step size ceiling / target step.
 * @return PreconditionResult holding optimal tau, reduced operator Q, and reduced initial state.
 */
PreconditionResult alg10_precondition(const Csr& D, const std::vector<double>& P0,
                                      int d, double eps, double t_eps);

/**
 * @brief Thesis Algorithm 11: Approximation Algorithm.
 *
 * Reference: Thesis Section 5.2 & Algorithm 11.
 * 1. Computes right eigenbasis Mnd of D for slowest decaying modes.
 * 2. Forms transformation T = [I, Tmd^T; 0, I] where Tmd = Tnd[d:, :].
 * 3. Constructs Mhat = RREF(T * Mnd).
 * 4. Reconstructs state distribution P_{k+1} = T^-1 * Mhat * Z_{k+1}.
 *
 * @param D Truncated transition matrix in CSR format.
 * @param precon Preconditioning result from Algorithm 10.
 * @param Z_next Solved reduced state vector Z(tau).
 * @return ApproximationResult holding reconstructed probability distribution and basis Mnd.
 */
ApproximationResult alg11_approximation(const Csr& D, const PreconditionResult& precon,
                                        const std::vector<double>& Z_next);

/**
 * @brief Thesis Algorithm 12: Transient Solution Function.
 *
 * Reference: Thesis Section 5.3 & Algorithm 12.
 * Evaluates the continuous probability distribution P(t) at any query time t \in [t0, tf]
 * by locating the corresponding interval snapshot, computing Z(t) = exp(Q * (t - t_start)) * Z0,
 * and projecting back through inverse transformation T^-1 * Mhat.
 *
 * @param input_list Vector of recorded interval snapshots from Algorithm 9.
 * @param full_states Union of all reachable states across the entire simulation.
 * @param query_t Target query time point.
 * @return Reconstructed probability distribution vector over full_states.
 */
std::vector<double> alg12_transient_solution(const std::vector<TransientStepSnapshot>& input_list,
                                             const std::vector<std::vector<int>>& full_states,
                                             double query_t);

/**
 * @brief Thesis Algorithm 9: Transient Multi-Time-Step Finite State Projection with Preconditioning.
 *
 * Reference: Thesis Chapter 5 & Algorithm 9.
 * Integrates state space expansion, preconditioning (Algorithm 10), reduced integration, and approximation
 * (Algorithm 11) to simulate stiff CME systems across [t0, tf].
 *
 * @param net Reaction network specification.
 * @param t0 Initial simulation time (typically 0.0).
 * @param tf Final simulation time.
 * @param eps Error tolerance.
 * @param t_eps Step size ceiling / target step.
 * @param d Reduced dimension for invariant eigenbasis.
 * @return PreconditionedFspResult containing full state trajectory snapshots and distribution at tf.
 */
PreconditionedFspResult alg9_preconditioned_fsp(const Network& net, double t0, double tf,
                                                double eps, double t_eps, int d);

}  // namespace cme
