# CME Execution and Monitoring Guide

This guide details how to run the Chemical Master Equation (CME) pipeline using `cme_cli`, interpret execution metrics in real-time, inspect generated artifacts, and verify outputs.

---

## 1. Quick Start

Ensure the CLI binary is built (`make` on macOS/Linux or `nmake -f Makefile.win` on Windows).

Run a standard CME calculation on a reaction network:
```bash
./build/cme_cli examples/michaelis_menten.net --d 4 --dump-matrix --dump-states --out mm
```

---

## 2. CLI Options and Execution Flags

```text
usage: cme_cli <network.net> [options]
```

| Flag | Type / Default | Description |
|---|---|---|
| `<network.net>` | Path (Required) | Reaction network file (defines species, reactions, conservation laws, initial counts). |
| `--d <N>` | Integer (default: `2`) | Number of basis vectors / eigenvalues to extract (max: `64`). |
| `--mode <slow\|fast>` | Enum (default: `slow`) | `slow`: eigenvalues closest to 0 (slowest decay modes); `fast`: largest $\|\lambda\|$. |
| `--cutoff <X>` | Float | **Chebyshev acceleration**. Required for stiff/large systems. Cutoff frequency separating wanted vs unwanted modes ($|\ln \varepsilon| / t_\varepsilon$). |
| `--degree <N>` | Integer (default: `100`) | Chebyshev polynomial filter degree. |
| `--backend <cpu\|gpu>` | Enum (default: `cpu`) | Computing backend (`cpu` uses OpenMP multi-threading; `gpu` uses CUDA cuSPARSE/cuBLAS). |
| `--tol <X>` | Float (default: `1e-10`) | Convergence tolerance: $\|K B - B K_{red}\|_\infty < \text{tol}$. |
| `--max-iter <N>` | Integer (default: `200000`) | Maximum outer iterations before halting. |
| `--inner <N>` | Integer (default: `4`) | Inner matrix products per normalization step (plain power mode). |
| `--shift <X>` | Float (default: Gershgorin) | Shift $\sigma \ge \rho(D)$, defaults to $2 \max |D_{ii}|$. |
| `--out <PREFIX>` | String (default: `out`) | Prefix for all generated output files. |
| `--dump-matrix` | Flag | Export the full sparse transition matrix $D$ in Matrix Market format (`.D.mtx`). |
| `--dump-states` | Flag | Export state space enumeration to CSV (`.states.csv`). |
| `--verbose` | Flag | Print per-iteration progress metrics. |

---

## 3. How to Monitor Progress

### Standard Output Summary
When running `cme_cli`, the console outputs real-time phase transitions and performance stats:

```text
states n = 10   nnz = 34   (state space 0.004s, matrix 0.001s)
backend cpu, mode slow, d = 4, sigma = 18000
iterations 50  SpMM products 206  converged yes  residual 2.25e-12  iterate time 0.052s
```

### Key Metrics to Monitor
1. **State Space Assembly (`state space Xs, matrix Ys`)**:
   - `states n`: Total reachable discrete states found from the initial state via conservation constraints.
   - `nnz`: Number of non-zero entries in transition matrix $D$.
   - Should complete in seconds even for millions of states (e.g. 2M states in ~0.9s).
2. **Spectral Bound (`sigma`)**:
   - Defaults to Gershgorin circle bound $2 \max |D_{ii}|$. Must exceed the spectral radius $\rho(D)$.
3. **Iteration and Convergence**:
   - **`iterations`**: Number of outer iterations completed.
   - **`SpMM products`**: Total Sparse Matrix-Matrix multiplications performed.
   - **`converged`**: Must report `yes`. If `no`, the iteration reached `--max-iter` before reaching tolerance `--tol`.
   - **`residual`**: Invariant-subspace error:
     $$\max_{i,j} |(K B - B K_{red})_{ij}|$$
     Must be smaller than `--tol` (e.g. `< 1e-10`).
   - **`iterate time`**: Time spent inside the iterative eigensolver.

### Real-Time Monitoring with `--verbose`
Add `--verbose` to stream iteration steps, pivot changes, and residual decays:
```bash
./build/cme_cli examples/michaelis_menten.net --d 4 --verbose
```

---

## 4. Understanding Output Artifacts

Running with `--out <PREFIX>` produces the following files:

| Output File | Format | Contents |
|---|---|---|
| `<PREFIX>.basis.bin` | Binary (`float64`, row-major $n \times d$) | Invariant subspace basis matrix $B$, normalized so that $B[\text{pivots}, :] = I_d$. |
| `<PREFIX>.pivots.txt` | Text / ASCII integers | Indices of the $d$ pivot states chosen via Gauss-Jordan elimination. |
| `<PREFIX>.dreduced.csv` | CSV ($d \times d$) | Reduced transition matrix $D_{red}$ such that $D B \approx B D_{red}$. |
| `<PREFIX>.eigenvalues.csv` | CSV | Real and imaginary components of computed eigenvalues. |
| `<PREFIX>.eigenvectors.bin`| Binary (`float64`, $n \times d$) | Full state eigenvectors $B \cdot \text{eigvec}(D_{red})$ (produced when eigenvalues are all real). |
| `<PREFIX>.D.mtx` *(optional)* | Matrix Market (`.mtx`) | Full sparse transition matrix $D$ (generated when `--dump-matrix` is passed). |
| `<PREFIX>.states.csv` *(optional)* | CSV | Table of mapped states (counts per species per row; generated with `--dump-states`). |

---

## 5. Validating Results

A Python verification script is included in `utils/verify.py` to cross-check results against dense NumPy/SciPy eigensolvers (recommended for $n \le 6000$):

```bash
# 1. Run CME with matrix export
./build/cme_cli examples/michaelis_menten.net --d 4 --dump-matrix --out mm

# 2. Run verification
python3 utils/verify.py mm --d 4
```

**Expected output:**
```text
reference eigenvalues: [ -0.      -7.22123 -12.7483  -18.06454]
computed  eigenvalues: [ -0.      -7.22123 -12.7483  -18.06454]
max eigenvalue error : 1.24e-12
subspace residual    : 2.25e-12
identity at pivots   : True
```

---

## 6. Running Large or Stiff Networks

For stiff systems where reaction rate constants differ by orders of magnitude (e.g., `michaelis_menten_large.net` with $n \ge 231$):
* Standard power iteration may require $10^5 - 10^7$ matrix products.
* **Use Chebyshev acceleration** by setting `--cutoff`:

```bash
./build/cme_cli examples/michaelis_menten_large.net --d 4 --cutoff 140 --out mm_large
```

* `--cutoff <X>` defines the spectral threshold between target eigenvalues and the stiff decay spectrum ($|\ln \varepsilon| / t_\varepsilon$).
