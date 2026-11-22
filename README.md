# Dimensionality Reduction of the Chemical Master Equation

> **Interactive Walkthrough & Live Demo (GitHub Pages):**  
> [https://dr-midhun.github.io/CME-Dimensionality-Reduction/](https://dr-midhun.github.io/CME-Dimensionality-Reduction/)

C++17 / OpenMP / CUDA implementation of the Chemical Master Equation (CME) pipeline and dimensionality reduction algorithms based on the Ph.D. thesis:
> **Dimensionality Reduction of the Chemical Master Equation**  
> Midhun Kathanaruparambil Sukumaran (University of Waterloo, 2019)  
> Thesis reference: https://uwspace.uwaterloo.ca/items/173f4625-6e7f-4e0a-9909-3ae0c67e10dd

| Thesis | Here | Backends |
|---|---|---|
| Alg. 1 – iterative breadth-first CME generation (Kan et al.) | `src/alg1_cme.cpp` | CPU |
| Alg. 2 – state space (non-negative integer solutions of `C N = b`) | `src/alg2_state_space.cpp` | CPU (OpenMP) |
| Alg. 3 – transition matrix `D` | `src/alg3_transition.cpp` | CPU (OpenMP) |
| Alg. 4 – reduction algorithm (reduced IVP, projection & reconstruction) | `src/alg4_reduction.cpp` | CPU (OpenMP) |
| Alg. 5 – semi-orthogonal eigenbasis (block power iteration + pivoting) | `src/alg5_eigenbasis_cpu.cpp`, `src/alg5_eigenbasis_cuda.cpp` | CPU, GPU (cuSPARSE + cuBLAS) |
| Alg. 6 – classic Finite State Projection (Munsky & Khammash) | `src/alg6_7_8_fsp.cpp` | CPU |
| Alg. 7 – multiple time-step FSP with fixed time step | `src/alg6_7_8_fsp.cpp` | CPU |
| Alg. 8 – multiple time-step FSP with variable time step | `src/alg6_7_8_fsp.cpp` | CPU |
| Alg. 9 – transient multi-time-step FSP with preconditioning | `src/alg9_10_11_12_preconditioned_fsp.cpp` | CPU |
| Alg. 10 – preconditioning algorithm (optimal step & error dynamics) | `src/alg9_10_11_12_preconditioned_fsp.cpp` | CPU |
| Alg. 11 – approximation algorithm (right eigenbasis back-projection) | `src/alg9_10_11_12_preconditioned_fsp.cpp` | CPU |
| Alg. 12 – continuous transient solution evaluator | `src/alg9_10_11_12_preconditioned_fsp.cpp` | CPU |

States are stored as sorted 64-bit mixed-radix keys (8 B/state); neighbours are found by key
arithmetic + binary search. 2,003,001 states / 10 M nonzeros assemble in ~0.9 s on one core.

## Build

### Prerequisites / Init
To set up all prerequisites (compilers, OpenMP runtime, CMake) automatically:
* **macOS / Linux**:
  ```bash
  make init
  # or directly: ./utils/setup_prereqs.sh
  ```
* **Windows**:
  ```cmd
  nmake -f Makefile.win init
  # or directly: utils\setup_prereqs.bat
  ```

### Linux & macOS (Makefile)
* **Linux**: requires `g++` (or `clang++`) with OpenMP support.
* **macOS**: requires `libomp` (installed via `make init` or `brew install libomp`).
```bash
make                       # CPU build (auto-detects macOS + libomp / Linux)
make test                  # Run unit tests
make LAPACK=1              # + eigenvalues of the d x d reduced matrix (LAPACK_LIBS=-llapack)
make LAPACK=1 GPU=1        # + GPU backend (CUDA_HOME=/usr/local/cuda)
```

### Windows (MSVC)
Open the **x64 Native Tools Command Prompt for VS** and run either:
```cmd
nmake -f Makefile.win
nmake -f Makefile.win test
```
or use the batch build script:
```cmd
utils\build_win.bat
utils\build_win.bat test
```
*(Alternatively, you can build with MinGW64 `make` using `make CXX=g++`.)*

### Cross-platform (CMake)
```bash
cmake -B build
cmake --build build
ctest --test-dir build
```
*(On macOS, if CMake does not find OpenMP automatically: `cmake -B build -DOpenMP_ROOT=$(brew --prefix libomp)`)*

### Docker (Linux Container)
For an isolated, reproducible containerized Linux environment with OpenMP and LAPACK:
```bash
# Build Docker image
docker build -t cme .

# Run all unit and integration tests inside container
docker run --rm --entrypoint test_exhaustive cme

# Run CME CLI calculation with local volume mount
docker run --rm -v $(pwd):/cme/data -w /cme/data cme examples/michaelis_menten.net --d 4 --reduce --out out/mm

# Alternatively, via docker-compose or helper script:
docker compose up test
./utils/docker_run.sh cli examples/michaelis_menten.net --d 4 --reduce
```

## Use
```bash
./build/cme_cli examples/michaelis_menten.net --d 4 --dump-matrix --out mm
python3 utils/verify.py mm --d 4            # dense NumPy cross-check (small n)

# stiff / larger problems: Chebyshev filter, cutoff from thesis eq. 4.55  (|ln eps| / t_eps)
./build/cme_cli examples/michaelis_menten_large.net --d 4 --cutoff 140 --backend gpu
```
For a detailed breakdown of flags, metrics monitoring, and result validation, see **[docs/RUN_AND_MONITOR.md](docs/RUN_AND_MONITOR.md)**.

## Running Individual Thesis Algorithms (Algorithms 1–12)

Every algorithm described in the PhD thesis can be compiled, executed individually, and exported to structured output files using either dedicated `make` targets or the CLI flag `--alg <1-12>`:

| Algorithm | Thesis Name | Make Target | Key Output Files |
|---|---|---|---|
| **Alg 1** | Iterative CME Generation | `make run-alg1` | `out/alg1.states.csv`, `out/alg1.D.mtx` |
| **Alg 2** | State Space Generation | `make run-alg2` | `out/alg2.states.csv` |
| **Alg 3** | CME Transition Matrix Generation | `make run-alg3` | `out/alg3.D.mtx` |
| **Alg 4** | Dimensionality Reduction via Projection | `make run-alg4` | `out/alg4.q_reduced.csv`, `out/alg4.x0.csv`, `out/alg4.perm.txt`, `out/alg4.p_approx.csv` |
| **Alg 5** | Invariant Subspace Eigenbasis | `make run-alg5` | `out/alg5.basis.bin`, `out/alg5.pivots.txt`, `out/alg5.dreduced.csv`, `out/alg5.eigenvalues.csv` |
| **Alg 6** | Finite State Projection (Classic FSP) | `make run-alg6` | `out/alg6.fsp_prob.csv` |
| **Alg 7** | Multi-Step Fixed-Step FSP | `make run-alg7` | `out/alg7.fsp_prob.csv` |
| **Alg 8** | Multi-Step Variable-Step FSP | `make run-alg8` | `out/alg8.fsp_prob.csv` |
| **Alg 9** | Transient Preconditioned FSP | `make run-alg9` | `out/alg9.prefsp_prob.csv`, `out/alg9.prefsp_steps.csv` |
| **Alg 10** | Preconditioned Time Step Determination | `make run-alg10` | `out/alg10.tau.txt`, `out/alg10.q_reduced.csv`, `out/alg10.z0.csv`, `out/alg10.perm.txt` |
| **Alg 11** | Subinterval Approximation | `make run-alg11` | `out/alg11.p_next.csv` |
| **Alg 12** | Continuous Transient Solution | `make run-alg12` | `out/alg12.trajectory.csv` |

### Run all 12 algorithms sequentially:
```bash
make run-all-algs
```

### Custom Network and Output Options:
Override the target network file or output directory directly with `make`:
```bash
make run-alg4 NETWORK=examples/michaelis_menten.net OUT_DIR=out/custom
```

### Direct CLI Invocation:
```bash
# Run any individual algorithm by index (1 to 12):
./build/cme_cli examples/michaelis_menten.net --alg 1 --out out/alg1
./build/cme_cli examples/michaelis_menten.net --alg 4 --d 4 --solve-time 0.1 --out out/alg4
./build/cme_cli examples/michaelis_menten.net --alg 12 --d 4 --tf 0.5 --out out/alg12
```

### Docker Execution:
```bash
# Run individual algorithm in Docker:
./utils/docker_run.sh alg 4
./utils/docker_run.sh run-all-algs
```

Network files list species, reactions, conservation rows (`conserve`) and an initial state (see
`examples/`). Outputs: `basis.bin` (float64, row-major n×d, `B[pivots,:] = I`), `pivots.txt`,
`dreduced.csv` (D restricted to span B, so `D B = B Dred`), `eigenvalues.csv`, and
`eigenvectors.bin` (n×d, when all eigenvalues are real; they are `B · eigvec(Dred)`).

## Modes
* `--mode slow` (default): the d eigenvalues of `D` nearest 0 (`K = I + D/σ`, as in the thesis shift).
* `--mode fast`: the d largest |λ| (`K = D/σ`).
* `σ` defaults to the Gershgorin bound `2·max|D_ii|`; it must be ≥ the spectral radius of `D`.

## Deviations from the pseudo-code (please read)
1. **Convergence** is the invariant-subspace residual `max|K B − B Kred| < tol`, not the change
   between iterates (that can be tiny while the iteration is still far from converged).
2. Pivot/scale is done as `V ← V·inv(V[I,:])`; pivots come from Gauss–Jordan (the RREF) until they
   repeat, then are frozen, and re-selected automatically if the pivot block degrades.
3. **Chebyshev acceleration (`--cutoff`) is an addition, not in the thesis.** Plain power iteration
   needs ~σ/gap products; for stiff CMEs that is 10⁵–10⁷ (n = 231 did not converge in 3000
   iterations). The filter needs ~√(σ/gap). It assumes real or nearly real spectra (true for
   detailed-balance networks); `--cutoff` must separate wanted from unwanted eigenvalues and
   `--d` must equal the number wanted. A Ritz value outside the wanted region triggers a warning.

## Validation
`make test` reproduces the thesis Michaelis–Menten example (10 states, λ = 0, −7.2, −12.7, −18.1),
fast modes, and a 231-state Chebyshev case, all against dense NumPy/LAPACK. Chebyshev mode was also
checked at n = 861 and 5151 with `scripts/verify.py`.

**GPU status:** `src/eigenbasis_cuda.cpp` has been compiled only against stub headers and run through a
CPU mock of cuBLAS/cuSPARSE (`make test-gpu-logic`, same tests pass). It has **not** been run on a real
GPU, so check `./build/test_small gpu` first on your machine. Limits: `n·d < 2³¹`, nnz < 2³¹, d ≤ 64;
GPU memory ≈ 12 B/nnz + 3·n·d·8 B. Matrix generation is CPU-only.

## Cost caveat
Chebyshev work still scales like √(σ/gap)·log(1/tol). For the Michaelis–Menten family σ/gap ≈ 10⁵–10⁶,
so n = 5151 needed ~2.8·10⁵ SpMM products (44 s on one core). Shift-invert or Krylov–Schur would need
fewer products but a factorization or orthogonalization per step; not implemented.

## Thesis Reference
* Sukumaran, Midhun Kathanaruparambil. *Dimensionality Reduction of the Chemical Master Equation*. PhD thesis, University of Waterloo, 2019. [UWSpace Archive](https://uwspace.uwaterloo.ca/items/173f4625-6e7f-4e0a-9909-3ae0c67e10dd).
