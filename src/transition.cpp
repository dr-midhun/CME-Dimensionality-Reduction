// Algorithm 3 of the thesis: transition matrix of the CME.
//
// For every state N_v and reaction i: target N_v + S_i is located by key
// arithmetic + binary search (the thesis' index vector I), the propensity is
// c_i * prod_s nchoosek(N_s, R_is) (the thesis' C-hat), and the diagonal
// collects minus the total outflow.  Columns are assembled independently in
// parallel, then transposed to CSR (rows = target states).
#include "cme/transition.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace cme {
namespace {

inline double nchoosek(int n, int r) {
    if (r == 0) return 1.0;
    if (n < r) return 0.0;
    double v = 1.0;
    for (int j = 0; j < r; ++j) v = v * (n - j) / (j + 1);
    return v;
}

}  // namespace

Csr build_transition_matrix(const Network& net, const StateSpace& ss) {
    const int k = net.k();
    const int64_t n = ss.n();
    if (n >= 2147483647ll) throw std::runtime_error("too many states for 32-bit indices");

    // Reactions with a non-zero net change; precompute key shifts.
    std::vector<int> rx;
    std::vector<int64_t> shift;
    for (int i = 0; i < net.h(); ++i) {
        bool null = true;
        int64_t sh = 0;
        for (int s = 0; s < k; ++s) {
            if (net.S[i][s] != 0) null = false;
            sh += static_cast<int64_t>(net.S[i][s]) * static_cast<int64_t>(ss.stride[s]);
        }
        if (null) continue;
        rx.push_back(i);
        shift.push_back(sh);
    }
    const int h = static_cast<int>(rx.size());
    const int W = h + 1;  // max entries per column (transitions + diagonal)

    std::vector<int32_t> trow(static_cast<size_t>(n) * W);
    std::vector<double> tval(static_cast<size_t>(n) * W);
    std::vector<int32_t> tcnt(n);

    #pragma omp parallel
    {
        std::vector<int> N(k);
        std::vector<std::pair<int32_t, double>> ent;
        #pragma omp for schedule(static)
        for (int64_t v = 0; v < n; ++v) {
            const uint64_t key = ss.keys[v];
            ss.decode(key, N.data());
            ent.clear();
            double diag = 0.0;
            for (int q = 0; q < h; ++q) {
                const int i = rx[q];
                double prop = net.rate[i];
                for (int s = 0; s < k; ++s)
                    if (net.R[i][s] > 0) prop *= nchoosek(N[s], net.R[i][s]);
                if (prop == 0.0) continue;
                diag -= prop;
                bool inside = true;
                for (int s = 0; s < k; ++s) {
                    if (net.S[i][s] == 0) continue;
                    int t = N[s] + net.S[i][s];
                    if (t < 0 || t > ss.U[s]) { inside = false; break; }
                }
                if (!inside) continue;
                int64_t u = ss.find(static_cast<uint64_t>(static_cast<int64_t>(key) + shift[q]));
                if (u >= 0) ent.push_back({static_cast<int32_t>(u), prop});
            }
            ent.push_back({static_cast<int32_t>(v), diag});
            std::sort(ent.begin(), ent.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });
            // merge duplicate rows (two reactions with the same net change)
            int32_t cnt = 0;
            for (size_t e = 0; e < ent.size(); ++e) {
                if (cnt > 0 && trow[v * W + cnt - 1] == ent[e].first) {
                    tval[v * W + cnt - 1] += ent[e].second;
                } else {
                    trow[v * W + cnt] = ent[e].first;
                    tval[v * W + cnt] = ent[e].second;
                    ++cnt;
                }
            }
            tcnt[v] = cnt;
        }
    }

    // Transpose columns -> CSR rows (iterating columns ascending keeps columns sorted).
    int64_t nnz = 0;
    for (int64_t v = 0; v < n; ++v) nnz += tcnt[v];
    if (nnz >= 2147483647ll) throw std::runtime_error("nnz exceeds 2^31 (32-bit indices)");
    Csr D;
    D.n = static_cast<int32_t>(n);
    D.row_ptr.assign(n + 1, 0);
    for (int64_t v = 0; v < n; ++v)
        for (int32_t e = 0; e < tcnt[v]; ++e) ++D.row_ptr[trow[v * W + e] + 1];
    for (int64_t i = 0; i < n; ++i) D.row_ptr[i + 1] += D.row_ptr[i];
    D.col.resize(nnz);
    D.val.resize(nnz);
    std::vector<int32_t> pos(D.row_ptr.begin(), D.row_ptr.end() - 1);
    for (int64_t v = 0; v < n; ++v)
        for (int32_t e = 0; e < tcnt[v]; ++e) {
            int32_t u = trow[v * W + e];
            D.col[pos[u]] = static_cast<int32_t>(v);
            D.val[pos[u]] = tval[v * W + e];
            ++pos[u];
        }
    return D;
}

}  // namespace cme
