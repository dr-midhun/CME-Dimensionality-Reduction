#include "cme/alg9_10_11_12_preconditioned_fsp.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

#include "cme/alg3_transition.hpp"
#include "cme/alg4_reduction.hpp"
#include "cme/alg5_eigenbasis.hpp"
#include "cme/csr_util.hpp"
#include "cme/small_dense.hpp"

namespace cme {

namespace {

/**
 * @brief Computes binomial coefficient (n choose k) for reaction stoichiometry.
 *
 * @param n Available count of molecules.
 * @param k Required stoichiometric coefficient.
 * @return Value of n! / (k! * (n - k)!).
 */
double n_choose_k(int n, int k) {
    if (k < 0 || k > n) return 0.0;
    if (k == 0 || k == n) return 1.0;
    if (k == 1) return static_cast<double>(n);
    if (k == 2) return 0.5 * n * (n - 1);
    double res = 1.0;
    for (int i = 1; i <= k; ++i) res = res * (n - i + 1) / i;
    return std::round(res);
}

/**
 * @brief Evaluates mass-action reaction propensity for reaction r at state N.
 *
 * @param net Reaction network holding reaction rates and stoichiometric matrices.
 * @param r Reaction index.
 * @param N State vector of molecular populations.
 * @return Reaction transition rate.
 */
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

/**
 * @brief Constructs infinitesimal generator matrix D for a given sub-collection of states.
 *
 * Outflows to states outside this collection are treated as absorbing leakage,
 * ensuring column sums are less than or equal to zero.
 *
 * @param net Reaction network.
 * @param states Truncated state space collection.
 * @return Truncated infinitesimal generator matrix D in CSR format.
 */
Csr build_sub_D(const Network& net, const std::vector<std::vector<int>>& states) {
    const int32_t n = static_cast<int32_t>(states.size());
    const int h = net.m();
    // Build map for coordinate-to-index lookup
    std::map<std::vector<int>, int32_t> s_map;
    for (int32_t i = 0; i < n; ++i) s_map[states[i]] = i;

    // Accumulate sparse entries per row
    std::vector<std::map<int32_t, double>> row_entries(n);
    for (int32_t i = 0; i < n; ++i) row_entries[i][i] = 0.0;
    std::vector<double> outflow(n, 0.0);

    // Compute propensities from every state j
    for (int32_t j = 0; j < n; ++j) {
        const auto& s = states[j];
        for (int r = 0; r < h; ++r) {
            double p = reaction_propensity(net, r, s);
            if (p <= 0.0) continue;
            outflow[j] += p;

            // Compute target state coordinates
            std::vector<int> target(net.k());
            for (int k = 0; k < net.k(); ++k) target[k] = s[k] + net.S[r][k];
            auto it = s_map.find(target);
            if (it != s_map.end()) {
                int32_t i = it->second;
                if (i != j) row_entries[i][j] += p;
            }
        }
    }

    // Set diagonal elements to negative outflow
    for (int32_t j = 0; j < n; ++j) row_entries[j][j] = -outflow[j];

    // Build CSR structure
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

/**
 * @brief Expands local state collection by one reaction step along all reachable pathways.
 *
 * @param net Reaction network.
 * @param states Truncated state space collection, modified in place.
 */
void expand_sub_states(const Network& net, std::vector<std::vector<int>>& states) {
    const size_t sz = states.size();
    const int h = net.m();
    std::map<std::vector<int>, bool> existing;
    for (const auto& s : states) existing[s] = true;

    for (size_t idx = 0; idx < sz; ++idx) {
        // Copy state to prevent pointer invalidation on push_back
        std::vector<int> s = states[idx];
        for (int r = 0; r < h; ++r) {
            if (reaction_propensity(net, r, s) <= 0.0) continue;
            std::vector<int> target(net.k());
            for (int k = 0; k < net.k(); ++k) target[k] = s[k] + net.S[r][k];
            if (!existing[target]) {
                existing[target] = true;
                states.push_back(target);
            }
        }
    }
}

}  // namespace

/**
 * @brief Implements Thesis Algorithm 10: Preconditioning Algorithm.
 *
 * Forms reduced dynamical system using left invariant eigenbasis of D^T and augmented error dynamics
 * to determine the optimal step size tau satisfying error bound eps.
 */
PreconditionResult alg10_precondition(const Csr& D, const std::vector<double>& P0,
                                      int d, double eps, double t_eps) {
    const int64_t n = D.n;
    if (d >= n) d = static_cast<int>(n - 1);
    if (d < 1) d = 1;

    PreconditionResult res;
    res.d = d;

    // Step 1: Compute left eigenbasis of D by running Algorithm 5 on transposed matrix D^T
    Csr Dt = transpose_csr(D);
    EigenOptions opt_left;
    opt_left.d = d;
    opt_left.mode = Mode::Slow;
    opt_left.tol = 1e-10;
    EigenResult res_left = compute_eigenbasis(Dt, Backend::Cpu, opt_left);

    // Step 2: Build permutation vector I such that pivot states appear first
    std::vector<int64_t> perm(n);
    std::vector<bool> in_piv(n, false);
    for (int j = 0; j < d; ++j) {
        perm[j] = res_left.pivots[j];
        in_piv[res_left.pivots[j]] = true;
    }
    int64_t curr = d;
    for (int64_t i = 0; i < n; ++i) {
        if (!in_piv[i]) perm[curr++] = i;
    }
    std::vector<int64_t> perm_inv(n);
    for (int64_t i = 0; i < n; ++i) perm_inv[perm[i]] = i;

    res.perm = perm;

    // Step 3: Re-index left eigenbasis Tnd and initial state distribution P0
    res.Tnd_re.resize(static_cast<size_t>(n) * d);
    std::vector<double> P0_re(n);
    for (int64_t i = 0; i < n; ++i) {
        int64_t orig = perm[i];
        P0_re[i] = P0[orig];
        for (int c = 0; c < d; ++c) {
            res.Tnd_re[i * d + c] = res_left.basis[orig * d + c];
        }
    }

    // Step 4: Scale Tnd_re so its top d x d submatrix is the Identity matrix
    {
        std::vector<double> T_top(static_cast<size_t>(d) * d);
        for (int r = 0; r < d; ++r) {
            for (int c = 0; c < d; ++c) {
                T_top[r * d + c] = res.Tnd_re[r * d + c];
            }
        }
        dense_inverse(d, T_top.data());
        std::vector<double> T_scaled(static_cast<size_t>(n) * d);
        for (int64_t r = 0; r < n; ++r) {
            for (int c = 0; c < d; ++c) {
                double s = 0.0;
                for (int k = 0; k < d; ++k) s += res.Tnd_re[r * d + k] * T_top[k * d + c];
                T_scaled[r * d + c] = s;
            }
        }
        res.Tnd_re = std::move(T_scaled);
    }

    // Step 5: Construct reduced transition rate matrix Q_dd = Tnd_re^T * D_re[:, 0:d]
    const size_t dd = static_cast<size_t>(d) * d;
    res.Q.assign(dd, 0.0);
    for (int c = 0; c < d; ++c) {
        int64_t orig_c = perm[c];
        for (int32_t p = Dt.row_ptr[orig_c]; p < Dt.row_ptr[orig_c + 1]; ++p) {
            int64_t orig_k = Dt.col[p];
            int64_t re_k = perm_inv[orig_k];
            double val = Dt.val[p];
            for (int r = 0; r < d; ++r) {
                res.Q[r * d + c] += res.Tnd_re[re_k * d + r] * val;
            }
        }
    }

    // Step 6: Compute reduced initial state Z(0) = Tnd^T * P0
    res.Z0.assign(d, 0.0);
    for (int r = 0; r < d; ++r) {
        double s = 0.0;
        for (int64_t k = 0; k < n; ++k) s += res.Tnd_re[k * d + r] * P0_re[k];
        res.Z0[r] = s;
    }

    // Step 7: Form augmented error dynamical system (Thesis eq. 5.6):
    // [ \dot{Z} ; \dot{Z_eps} ] = [ Q, 0 ; -1^T Q, 0 ] [ Z ; Z_eps ]
    const int d_aug = d + 1;
    std::vector<double> Q_aug(static_cast<size_t>(d_aug) * d_aug, 0.0);
    for (int r = 0; r < d; ++r) {
        for (int c = 0; c < d; ++c) {
            Q_aug[r * d_aug + c] = res.Q[r * d + c];
        }
    }
    // Bottom row represents outflow: -1^T Q
    for (int c = 0; c < d; ++c) {
        double col_sum = 0.0;
        for (int r = 0; r < d; ++r) col_sum += res.Q[r * d + c];
        Q_aug[d * d_aug + c] = -col_sum;
    }

    std::vector<double> Z_aug0(d_aug);
    double sumZ0 = 0.0;
    for (int i = 0; i < d; ++i) {
        Z_aug0[i] = res.Z0[i];
        sumZ0 += res.Z0[i];
    }
    Z_aug0[d] = 1.0 - sumZ0;

    // Step 8: Search for optimal continuous time step tau where Z_eps(tau) <= eps
    double step_size = (t_eps > 0.0) ? t_eps : 0.01;
    double tau = step_size;
    std::vector<double> best_Z(d);

    for (int s = 1; s <= 20; ++s) {
        double test_t = s * (step_size / 4.0);
        std::vector<double> Q_t(static_cast<size_t>(d_aug) * d_aug);
        for (size_t i = 0; i < Q_t.size(); ++i) Q_t[i] = Q_aug[i] * test_t;
        std::vector<double> expQt(static_cast<size_t>(d_aug) * d_aug);
        matrix_exp(d_aug, Q_t.data(), expQt.data());

        // Propagate augmented state Z_aug(t) = exp(Q_aug * t) * Z_aug0
        std::vector<double> Z_aug_t(d_aug, 0.0);
        for (int r = 0; r < d_aug; ++r) {
            double sum_r = 0.0;
            for (int c = 0; c < d_aug; ++c) sum_r += expQt[r * d_aug + c] * Z_aug0[c];
            Z_aug_t[r] = sum_r;
        }

        double z_eps = std::fabs(Z_aug_t[d]);
        if (z_eps <= eps || s == 1) {
            tau = test_t;
            for (int i = 0; i < d; ++i) best_Z[i] = Z_aug_t[i];
        } else {
            break;
        }
    }

    res.tau = tau;
    res.Z_tau = best_Z;
    return res;
}

/**
 * @brief Implements Thesis Algorithm 11: Approximation Algorithm.
 *
 * Uses right eigenbasis Mnd of D and inverse coordinate transformation to reconstruct state probabilities.
 */
ApproximationResult alg11_approximation(const Csr& D, const PreconditionResult& precon,
                                        const std::vector<double>& Z_next) {
    const int64_t n = D.n;
    const int d = precon.d;

    ApproximationResult res;

    // Step 1: Compute right eigenbasis Mtemp of D
    EigenOptions opt_right;
    opt_right.d = d;
    opt_right.mode = Mode::Slow;
    opt_right.tol = 1e-10;
    EigenResult res_right = compute_eigenbasis(D, Backend::Cpu, opt_right);

    // Step 2: Re-index right eigenbasis using the permutation from preconditioning
    std::vector<double> Mtemp_re(static_cast<size_t>(n) * d);
    for (int64_t i = 0; i < n; ++i) {
        int64_t orig = precon.perm[i];
        for (int c = 0; c < d; ++c) {
            Mtemp_re[i * d + c] = res_right.basis[orig * d + c];
        }
    }

    // Step 3: Apply transformation T = [Idd, Tmd^T; 0, Imm] to Mtemp (Thesis eq. 5.12)
    std::vector<double> TM(static_cast<size_t>(n) * d);
    for (int r = 0; r < d; ++r) {
        for (int c = 0; c < d; ++c) {
            double s = Mtemp_re[r * d + c];
            for (int64_t m = d; m < n; ++m) {
                s += precon.Tnd_re[m * d + r] * Mtemp_re[m * d + c];
            }
            TM[r * d + c] = s;
        }
    }
    for (int64_t r = d; r < n; ++r) {
        for (int c = 0; c < d; ++c) {
            TM[r * d + c] = Mtemp_re[r * d + c];
        }
    }

    // Step 4: Normalize top d x d block of TM to Identity: Mnd = TM * inv(TM[0:d, :])
    std::vector<double> TM_top(static_cast<size_t>(d) * d);
    for (int r = 0; r < d; ++r) {
        for (int c = 0; c < d; ++c) {
            TM_top[r * d + c] = TM[r * d + c];
        }
    }
    dense_inverse(d, TM_top.data());

    res.Mnd_re.resize(static_cast<size_t>(n) * d);
    for (int64_t r = 0; r < n; ++r) {
        for (int c = 0; c < d; ++c) {
            double s = 0.0;
            for (int k = 0; k < d; ++k) s += TM[r * d + k] * TM_top[k * d + c];
            res.Mnd_re[r * d + c] = s;
        }
    }

    // Step 5: Evaluate approximate distribution P_{next} = T^-1 * Mnd_re * Z_{next}
    std::vector<double> Y(n, 0.0);
    for (int64_t r = 0; r < n; ++r) {
        double s = 0.0;
        for (int c = 0; c < d; ++c) s += res.Mnd_re[r * d + c] * Z_next[c];
        Y[r] = s;
    }
    // Inverse transformation T^-1 = [I, -Tmd^T; 0, I]
    std::vector<double> P_re(n, 0.0);
    for (int r = 0; r < d; ++r) {
        double s = Y[r];
        for (int64_t m = d; m < n; ++m) {
            s -= precon.Tnd_re[m * d + r] * Y[m];
        }
        P_re[r] = s;
    }
    for (int64_t r = d; r < n; ++r) P_re[r] = Y[r];

    // Map re-indexed distribution back to original indexing: P[perm[i]] = P_re[i]
    res.P_next.resize(n);
    for (int64_t i = 0; i < n; ++i) {
        res.P_next[precon.perm[i]] = P_re[i];
    }

    return res;
}

/**
 * @brief Implements Thesis Algorithm 12: Transient Solution Function.
 *
 * Evaluates the continuous probability distribution at target time query_t from recorded interval snapshots.
 */
std::vector<double> alg12_transient_solution(const std::vector<TransientStepSnapshot>& input_list,
                                             const std::vector<std::vector<int>>& full_states,
                                             double query_t) {
    const size_t n_full = full_states.size();
    std::vector<double> P_sol(n_full, 0.0);
    if (input_list.empty()) return P_sol;

    // Locate snapshot interval containing query_t
    const TransientStepSnapshot* target_step = &input_list.back();
    for (const auto& step : input_list) {
        if (query_t >= step.t_start && query_t <= step.t_end) {
            target_step = &step;
            break;
        }
    }

    // Relative elapsed time inside interval
    double delta_t = query_t - target_step->t_start;
    if (delta_t < 0.0) delta_t = 0.0;

    // Evaluate reduced state Z(delta_t) = exp(Q * delta_t) * Z0
    int d = target_step->d;
    std::vector<double> Qt(static_cast<size_t>(d) * d);
    for (size_t i = 0; i < Qt.size(); ++i) Qt[i] = target_step->Q[i] * delta_t;
    std::vector<double> expQt(static_cast<size_t>(d) * d);
    matrix_exp(d, Qt.data(), expQt.data());

    std::vector<double> Z_t(d, 0.0);
    for (int r = 0; r < d; ++r) {
        double s = 0.0;
        for (int c = 0; c < d; ++c) s += expQt[r * d + c] * target_step->Z0[c];
        Z_t[r] = s;
    }

    // Local reconstructed state: P_local = T^-1 * M * Z(delta_t)
    const int64_t n_loc = static_cast<int64_t>(target_step->states.size());
    std::vector<double> Y(n_loc, 0.0);
    for (int64_t r = 0; r < n_loc; ++r) {
        double s = 0.0;
        for (int c = 0; c < d; ++c) s += target_step->Mnd_re[r * d + c] * Z_t[c];
        Y[r] = s;
    }
    std::vector<double> P_loc_re(n_loc, 0.0);
    for (int r = 0; r < d; ++r) {
        double s = Y[r];
        for (int64_t m = d; m < n_loc; ++m) {
            s -= target_step->Tnd_re[m * d + r] * Y[m];
        }
        P_loc_re[r] = s;
    }
    for (int64_t r = d; r < n_loc; ++r) P_loc_re[r] = Y[r];

    // Permute back to local state order
    std::vector<double> P_loc(n_loc);
    for (int64_t i = 0; i < n_loc; ++i) P_loc[target_step->perm[i]] = P_loc_re[i];

    // Map local probabilities into global state vector Nfull
    std::map<std::vector<int>, int64_t> full_map;
    for (size_t i = 0; i < n_full; ++i) full_map[full_states[i]] = static_cast<int64_t>(i);

    for (int64_t i = 0; i < n_loc; ++i) {
        auto it = full_map.find(target_step->states[i]);
        if (it != full_map.end()) {
            P_sol[it->second] = P_loc[i];
        }
    }

    return P_sol;
}

/**
 * @brief Implements Thesis Algorithm 9: Transient Multi-Time-Step FSP with Preconditioning.
 */
PreconditionedFspResult alg9_preconditioned_fsp(const Network& net, double t0, double tf,
                                                double eps, double t_eps, int d) {
    PreconditionedFspResult res;
    res.tf = tf;

    // Seed state space with starting state x0
    std::vector<std::vector<int>> Nk = {net.x0};
    expand_sub_states(net, Nk);
    std::vector<double> Pk(Nk.size(), 0.0);
    Pk[0] = 1.0;

    // Track full state space union
    std::map<std::vector<int>, bool> full_set;
    for (const auto& s : Nk) full_set[s] = true;

    double tk = t0;
    int k = 0;

    // Outer time integration loop across [t0, tf]
    while (tk < tf && k < 500) {
        double tau_k = 0.0;
        Csr Dk;
        PreconditionResult precon;

        // Thesis Algorithm 9 lines 10-14:
        // Expand state space iteratively if valid time step tau_k < t_eps
        int inner_exp = 0;
        while (tau_k < t_eps && inner_exp < 50) {
            Dk = build_sub_D(net, Nk);
            precon = alg10_precondition(Dk, Pk, d, eps, t_eps);
            tau_k = precon.tau;

            if (tau_k < t_eps && tk + tau_k < tf) {
                size_t old_sz = Nk.size();
                expand_sub_states(net, Nk);
                if (Nk.size() > old_sz) {
                    Pk.resize(Nk.size(), 0.0);
                } else {
                    break;
                }
                ++inner_exp;
            } else {
                break;
            }
        }

        // Clamp step to final time tf if remaining duration is shorter
        if (tk + tau_k > tf) {
            tau_k = tf - tk;
            // Recompute Z_tau at the clamped final step tau_k = tf - tk
            std::vector<double> Qt(static_cast<size_t>(precon.d) * precon.d);
            for (size_t i = 0; i < Qt.size(); ++i) Qt[i] = precon.Q[i] * tau_k;
            std::vector<double> expQt(static_cast<size_t>(precon.d) * precon.d);
            matrix_exp(precon.d, Qt.data(), expQt.data());
            precon.Z_tau.assign(precon.d, 0.0);
            for (int r = 0; r < precon.d; ++r) {
                double s = 0.0;
                for (int c = 0; c < precon.d; ++c) s += expQt[r * precon.d + c] * precon.Z0[c];
                precon.Z_tau[r] = s;
            }
        }
        if (tau_k <= 0.0) break;

        // Step 15: Compute state approximation using Algorithm 11
        ApproximationResult approx = alg11_approximation(Dk, precon, precon.Z_tau);

        // Record subinterval snapshot for continuous transient reconstruction (Algorithm 12)
        TransientStepSnapshot snap;
        snap.t_start = tk;
        snap.t_end = tk + tau_k;
        snap.tau = tau_k;
        snap.d = precon.d;
        snap.perm = precon.perm;
        snap.states = Nk;
        snap.Tnd_re = precon.Tnd_re;
        snap.Mnd_re = approx.Mnd_re;
        snap.Q = precon.Q;
        snap.Z0 = precon.Z0;
        res.steps.push_back(snap);

        // Advance simulation time and update state probability
        tk += tau_k;
        Pk = approx.P_next;
        ++k;

        for (const auto& s : Nk) full_set[s] = true;
    }

    // Assemble global union of states Nfull
    for (const auto& kv : full_set) res.full_states.push_back(kv.first);
    // Evaluate continuous transient solution at tf
    res.P_tf = alg12_transient_solution(res.steps, res.full_states, tf);
    res.num_subintervals = k;
    res.converged = (tk >= tf - 1e-7);
    return res;
}

}  // namespace cme
