// Algorithm 2 of the thesis: all non-negative integer solutions of C N = b.
//
// Species are ordered by decreasing upper bound U, C is brought to RREF, the
// species at pivot columns ("dependent") are obtained from the non-pivot
// ("independent") ones, which are enumerated by depth-first search with the
// pruning rule  C_nonpivot * N_nonpivot <= b.  The thesis builds the
// candidate matrix explicitly; the DFS visits the same candidates without
// materialising them, and the first level is split across OpenMP threads.
#include "cme/state_space.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace cme {

int64_t StateSpace::find(uint64_t key) const {
    auto it = std::lower_bound(keys.begin(), keys.end(), key);
    if (it == keys.end() || *it != key) return -1;
    return it - keys.begin();
}

namespace {

struct Problem {
    int k, r0;                         // species, rows of C
    std::vector<std::vector<int>> C;   // original (non-negative) rows
    std::vector<double> b;
    std::vector<int> U;
    std::vector<uint64_t> stride;
    // RREF in permuted column order
    std::vector<int> perm;             // perm[pos] = species index
    std::vector<std::vector<double>> A;  // rank x (k+1)
    std::vector<int> pivot_pos, free_pos;  // positions (in perm order)
};

struct Ctx {
    const Problem* P;
    std::vector<int> N;                // species values for free species
    std::vector<double> partial;       // C_free * N_free so far
    std::vector<uint64_t>* out;
};

void leaf(Ctx& c) {
    const Problem& P = *c.P;
    std::vector<int> full(P.k, 0);
    for (int pos : P.free_pos) full[P.perm[pos]] = c.N[P.perm[pos]];
    const int rank = static_cast<int>(P.pivot_pos.size());
    for (int t = 0; t < rank; ++t) {
        double x = P.A[t][P.k];
        for (int pos : P.free_pos) x -= P.A[t][pos] * c.N[P.perm[pos]];
        double xr = std::nearbyint(x);
        if (std::fabs(x - xr) > 1e-6 || xr < 0) return;
        full[P.perm[P.pivot_pos[t]]] = static_cast<int>(xr);
    }
    uint64_t key = 0;
    for (int s = 0; s < P.k; ++s) {
        if (full[s] > P.U[s]) return;
        key += static_cast<uint64_t>(full[s]) * P.stride[s];
    }
    c.out->push_back(key);
}

void dfs(Ctx& c, size_t level) {
    const Problem& P = *c.P;
    if (level == P.free_pos.size()) { leaf(c); return; }
    int s = P.perm[P.free_pos[level]];
    std::vector<double> saved = c.partial;
    for (int v = 0; v <= P.U[s]; ++v) {
        bool ok = true;
        for (int i = 0; i < P.r0; ++i) {
            c.partial[i] = saved[i] + static_cast<double>(P.C[i][s]) * v;
            if (c.partial[i] > P.b[i] + 1e-9) { ok = false; break; }
        }
        if (!ok) break;  // C >= 0 so larger v only violates more
        c.N[s] = v;
        dfs(c, level + 1);
    }
    c.partial = saved;
    c.N[s] = 0;
}

}  // namespace

StateSpace generate_state_space(const Network& net) {
    Problem P;
    P.k = net.k();
    P.C = net.C;
    P.r0 = static_cast<int>(net.C.size());
    P.b.assign(P.r0, 0.0);
    for (int i = 0; i < P.r0; ++i)
        for (int s = 0; s < P.k; ++s) P.b[i] += static_cast<double>(net.C[i][s]) * net.x0[s];

    // Upper bounds U_s = floor(min_i b_i / C_is).
    P.U.assign(P.k, -1);
    for (int s = 0; s < P.k; ++s) {
        double best = -1;
        for (int i = 0; i < P.r0; ++i)
            if (P.C[i][s] > 0) {
                double u = std::floor(P.b[i] / P.C[i][s] + 1e-9);
                best = (best < 0) ? u : std::min(best, u);
            }
        if (best < 0) throw std::runtime_error("species '" + net.species[s] + "' appears in no conservation law: state space unbounded");
        P.U[s] = static_cast<int>(best);
    }

    // Mixed-radix strides with overflow check.
    P.stride.assign(P.k, 1);
    unsigned __int128 acc = 1;
    for (int s = 0; s < P.k; ++s) {
        P.stride[s] = static_cast<uint64_t>(acc);
        acc *= static_cast<unsigned __int128>(P.U[s]) + 1;
        if (acc > (static_cast<unsigned __int128>(1) << 63))
            throw std::runtime_error("state key overflows 64 bits (too many species / too large bounds)");
    }

    // Order species by decreasing bound; RREF of [C | b].
    P.perm.resize(P.k);
    std::iota(P.perm.begin(), P.perm.end(), 0);
    std::stable_sort(P.perm.begin(), P.perm.end(), [&](int a, int b) { return P.U[a] > P.U[b]; });
    std::vector<std::vector<double>> M(P.r0, std::vector<double>(P.k + 1));
    for (int i = 0; i < P.r0; ++i) {
        for (int pos = 0; pos < P.k; ++pos) M[i][pos] = P.C[i][P.perm[pos]];
        M[i][P.k] = P.b[i];
    }
    int row = 0;
    std::vector<char> is_pivot(P.k, 0);
    for (int col = 0; col < P.k && row < P.r0; ++col) {
        int p = row;
        for (int i = row + 1; i < P.r0; ++i)
            if (std::fabs(M[i][col]) > std::fabs(M[p][col])) p = i;
        if (std::fabs(M[p][col]) < 1e-9) continue;
        std::swap(M[p], M[row]);
        double piv = M[row][col];
        for (double& x : M[row]) x /= piv;
        for (int i = 0; i < P.r0; ++i) {
            if (i == row) continue;
            double f = M[i][col];
            if (f == 0.0) continue;
            for (int j = 0; j <= P.k; ++j) M[i][j] -= f * M[row][j];
        }
        P.pivot_pos.push_back(col);
        is_pivot[col] = 1;
        ++row;
    }
    M.resize(row);
    P.A = M;
    for (int pos = 0; pos < P.k; ++pos)
        if (!is_pivot[pos]) P.free_pos.push_back(pos);

    // Enumerate. Level 0 is parallelised.
    std::vector<uint64_t> all;
    if (P.free_pos.empty()) {
        std::vector<uint64_t> out;
        Ctx c{&P, std::vector<int>(P.k, 0), std::vector<double>(P.r0, 0.0), &out};
        leaf(c);
        all = out;
    } else {
        int s0 = P.perm[P.free_pos[0]];
        int top = P.U[s0];
        std::vector<std::vector<uint64_t>> per(top + 1);
        #pragma omp parallel for schedule(dynamic)
        for (int v = 0; v <= top; ++v) {
            Ctx c{&P, std::vector<int>(P.k, 0), std::vector<double>(P.r0, 0.0), &per[v]};
            bool ok = true;
            for (int i = 0; i < P.r0; ++i) {
                c.partial[i] = static_cast<double>(P.C[i][s0]) * v;
                if (c.partial[i] > P.b[i] + 1e-9) ok = false;
            }
            if (!ok) continue;
            c.N[s0] = v;
            dfs(c, 1);
        }
        size_t total = 0;
        for (auto& v : per) total += v.size();
        all.reserve(total);
        for (auto& v : per) all.insert(all.end(), v.begin(), v.end());
    }
    std::sort(all.begin(), all.end());
    all.erase(std::unique(all.begin(), all.end()), all.end());
    if (all.size() >= 2147483647ull) throw std::runtime_error("state space exceeds 2^31 states (32-bit indices)");

    StateSpace ss;
    ss.k = P.k;
    ss.U = P.U;
    ss.stride = P.stride;
    ss.keys = std::move(all);
    return ss;
}

}  // namespace cme
