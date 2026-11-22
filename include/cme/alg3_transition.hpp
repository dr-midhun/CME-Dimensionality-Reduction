#pragma once
#include "cme/csr.hpp"
#include "cme/network.hpp"
#include "cme/state_space.hpp"

namespace cme {

// Transition matrix D of the CME (thesis Algorithm 3), dp/dt = D p.
// D[u][v] = propensity of the transition  N_v -> N_u ; D[v][v] = -total outflow.
// Every row/column has a (possibly zero) diagonal entry stored explicitly.
Csr build_transition_matrix(const Network& net, const StateSpace& ss);

}  // namespace cme
