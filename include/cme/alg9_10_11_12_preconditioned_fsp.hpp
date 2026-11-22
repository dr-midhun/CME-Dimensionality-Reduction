#pragma once
#include <vector>
#include "cme/csr.hpp"
#include "cme/network.hpp"

namespace cme {

// Preconditioning result (Algorithm 10)
struct PreconditionResult {
    Csr D;                           // re-indexed transition matrix
    int d = 0;                       // reduced dimension
    std::vector<int64_t> perm;       // permutation vector I
    std::vector<double> Tnd_re;      // left semi-orthogonal eigenbasis (size n x d, row-major)
    std::vector<double> Q;           // reduced generator (size d x d, row-major)
    std::vector<double> Z0;          // reduced initial condition (size d)
    double tau = 0.0;                // optimal time-step identified
    std::vector<double> Z_tau;       // solution Z(tau) at time tau
};

// Approximation result (Algorithm 11)
struct ApproximationResult {
    std::vector<double> P_next;      // approximate solution vector at next time point
    std::vector<double> Mnd_re;      // right eigenbasis Mnd (size n x d, row-major)
};

// Step snapshot for transient solution accumulation (Algorithm 12)
struct TransientStepSnapshot {
    double t_start = 0.0;
    double t_end = 0.0;
    double tau = 0.0;
    int d = 0;
    std::vector<int64_t> perm;
    std::vector<std::vector<int>> states; // state space Nk in this interval
    std::vector<double> Tnd_re;          // Tk
    std::vector<double> Mnd_re;          // Mk
    std::vector<double> Q;               // reduced generator
    std::vector<double> Z0;              // initial reduced vector
};

struct PreconditionedFspResult {
    double tf = 0.0;
    std::vector<std::vector<int>> full_states; // Nfull: union of all reached states
    std::vector<double> P_tf;                  // approximate probability distribution at tf over Nfull
    std::vector<TransientStepSnapshot> steps;  // snapshot history for Algorithm 12
    int num_subintervals = 0;
    bool converged = false;
};

// Thesis Algorithm 10: Preconditioning Algorithm
// Computes left eigenbasis, re-indexes, forms augmented error ODE system [Q 0; -1^T Q 0],
// and determines the optimal valid time step tau where error Z_eps(t) <= eps.
PreconditionResult alg10_precondition(const Csr& D, const std::vector<double>& P0,
                                      int d, double eps, double t_eps);

// Thesis Algorithm 11: Approximation Algorithm
// Computes right eigenbasis Mnd, applies transformation T = [I Tmd^T; 0 I], and evaluates P_{k+1} = T^-1 Mnd Z_{k+1}.
ApproximationResult alg11_approximation(const Csr& D, const PreconditionResult& precon,
                                        const std::vector<double>& Z_next);

// Thesis Algorithm 12: Transient Solution Function
// Evaluates the continuous transient solution P(t) across all subintervals in (t0, tf).
std::vector<double> alg12_transient_solution(const std::vector<TransientStepSnapshot>& input_list,
                                             const std::vector<std::vector<int>>& full_states,
                                             double query_t);

// Thesis Algorithm 9: Transient Multi-Time-Step Finite State Projection with Preconditioning
// Integrates state expansion, preconditioning (Alg 10), reduced integration, and approximation (Alg 11) across [t0, tf].
PreconditionedFspResult alg9_preconditioned_fsp(const Network& net, double t0, double tf,
                                                double eps, double t_eps, int d);

}  // namespace cme
