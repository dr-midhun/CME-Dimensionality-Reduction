#!/usr/bin/env python3
"""Cross-check cme_cli output against dense NumPy (small n only).

  cme_cli net.net --d 4 --out out --dump-matrix
  python3 scripts/verify.py out --d 4 [--mode slow|fast]
"""
import argparse
import warnings

import numpy as np
from scipy.io import mmread

warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("prefix")
ap.add_argument("--d", type=int, required=True)
ap.add_argument("--mode", default="slow", choices=["slow", "fast"])
a = ap.parse_args()

D = mmread(a.prefix + ".D.mtx").toarray()
n = D.shape[0]
if n > 6000:
    raise SystemExit("matrix too large for dense verification; use scipy.sparse.linalg.eigs instead")
B = np.fromfile(a.prefix + ".basis.bin", dtype=np.float64).reshape(n, a.d)
Dred = np.loadtxt(a.prefix + ".dreduced.csv", delimiter=",").reshape(a.d, a.d)

w = np.linalg.eigvals(D)
order = np.argsort(-w.real) if a.mode == "slow" else np.argsort(-np.abs(w))
ref = w[order][: a.d]
got = np.linalg.eigvals(Dred)
got = got[np.argsort(-got.real)] if a.mode == "slow" else got[np.argsort(-np.abs(got))]

print("reference eigenvalues:", np.round(ref, 6))
print("computed  eigenvalues:", np.round(got, 6))
print("max eigenvalue error :", np.max(np.abs(np.sort_complex(ref) - np.sort_complex(got))))
print("subspace residual    :", np.max(np.abs(D @ B - B @ Dred)) / max(1.0, np.max(np.abs(D @ B))))
piv = np.loadtxt(a.prefix + ".pivots.txt", dtype=int).reshape(-1)
print("identity at pivots   :", np.allclose(B[piv, :], np.eye(a.d)))
