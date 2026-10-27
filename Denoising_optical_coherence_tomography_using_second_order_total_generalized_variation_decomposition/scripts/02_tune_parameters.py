"""
Step 4 (parameter selection): coarse-to-fine grid search for alpha0, alpha1, beta and the
split-Bregman penalties lambda1..3.

Protocol (stated in the report, chosen to avoid per-experiment tuning):

    * tuning is done on the synthetic phantom at ONE noise level, variance 0.04
    * the objective is PSNR (ties broken by SSIM)
    * stage 1: (alpha1, beta, r = alpha0/alpha1) with lambda = (2,2,2)
    * stage 2: lambda1 = lambda2 = lambda3 over a logarithmic range at the stage-1 optimum
    * stage 3: asymmetric penalties around the stage-2 optimum
    * the final values are then FROZEN and used for every other noise level, every real
      image and every figure

Output: results/tuning/grid_stage{1,2,3}.csv  (+ a summary JSON)
"""

from __future__ import annotations

import csv
import itertools
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.metrics import all_metrics                     # noqa: E402
from src.phantom import add_gaussian_noise, synthetic_oct_phantom  # noqa: E402
from src.tgv import tgv_decompose                       # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "tuning")
os.makedirs(OUT, exist_ok=True)

VAR = 0.04          # tuning noise level
MAX_ITER = 300


def run(clean, noisy, a0, a1, beta, l1, l2, l3, ramp=1.0):
    t0 = time.perf_counter()
    u, _ = tgv_decompose(noisy, a0, a1, beta, l1, l2, l3, max_iter=MAX_ITER, tol=1e-3)
    dt = time.perf_counter() - t0
    m = all_metrics(clean, u)
    m.update(dict(alpha0=a0, alpha1=a1, beta=beta, lam1=l1, lam2=l2, lam3=l3,
                  psnr_ramp=m["PSNR"], **{"time_s": dt}))
    return m


def main():
    clean = synthetic_oct_phantom()
    noisy = add_gaussian_noise(clean, VAR, seed=1)
    base = all_metrics(clean, noisy)
    print(f"noisy baseline @ var {VAR}: {base}")

    # ---------------- stage 1 -------------------------------------------------------
    rows = []
    for a1, beta, r in itertools.product([0.02, 0.05, 0.1, 0.2, 0.4],
                                         [0.005, 0.02, 0.05, 0.1, 0.2],
                                         [0.5, 1.0, 2.0]):
        a0 = r * a1
        m = run(clean, noisy, a0, a1, beta, 2.0, 2.0, 2.0)
        rows.append(m)
        print(f"  a1={a1:<5} a0={a0:<6.3f} beta={beta:<5} -> "
              f"PSNR {m['PSNR']:.2f} SSIM {m['SSIM']:.4f}")
    with open(os.path.join(OUT, "grid_stage1.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    best1 = max(rows, key=lambda r: (r["PSNR"], r["SSIM"]))
    print("stage-1 best:", {k: best1[k] for k in
                            ("alpha0", "alpha1", "beta", "PSNR", "SSIM")})

    # ---------------- stage 2: penalties -------------------------------------------
    rows2 = []
    for lam in [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
        m = run(clean, noisy, best1["alpha0"], best1["alpha1"], best1["beta"],
                lam, lam, lam)
        rows2.append(m)
        print(f"  lam={lam:<5} -> PSNR {m['PSNR']:.2f} SSIM {m['SSIM']:.4f} "
              f"({m['time_s']:.1f}s)")
    with open(os.path.join(OUT, "grid_stage2.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows2[0].keys()))
        w.writeheader()
        w.writerows(rows2)
    best2 = max(rows2, key=lambda r: (r["PSNR"], r["SSIM"]))
    lam0 = best2["lam1"]
    print("stage-2 best lam:", lam0)

    # ---------------- stage 3: asymmetric penalties --------------------------------
    rows3 = []
    base_cfg = (best1["alpha0"], best1["alpha1"], best1["beta"])
    for l1, l2, l3 in itertools.product([lam0 / 4, lam0, lam0 * 4], repeat=3):
        m = run(clean, noisy, *base_cfg, l1, l2, l3)
        rows3.append(m)
        print(f"  lam=({l1:g},{l2:g},{l3:g}) -> PSNR {m['PSNR']:.2f} SSIM {m['SSIM']:.4f}")
    with open(os.path.join(OUT, "grid_stage3.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows3[0].keys()))
        w.writeheader()
        w.writerows(rows3)
    best3 = max(rows3, key=lambda r: (r["PSNR"], r["SSIM"]))
    print("stage-3 best:", {k: best3[k] for k in
                            ("lam1", "lam2", "lam3", "PSNR", "SSIM")})

    # ---------------- penalty invariance (fixed point is lambda-independent) --------
    # With exact convergence the split-Bregman penalties do not change the minimiser of
    # (2.11); they only change the conditioning and hence the iteration count.  Verify
    # this so that the stage-2/3 spread can be attributed to the stopping tolerance.
    inv = []
    for lam in [(2.0, 2.0, 2.0), (0.5, 0.5, 0.5), (best3["lam1"], best3["lam2"],
                                                    best3["lam3"])]:
        u, _, info = tgv_decompose(noisy, best1["alpha0"], best1["alpha1"],
                                   best1["beta"], *lam, max_iter=4000, tol=1e-5,
                                   return_info=True)
        m = all_metrics(clean, u)
        inv.append(dict(lam1=lam[0], lam2=lam[1], lam3=lam[2],
                        iters=int(info["n_iter"]), psnr=m["PSNR"], ssim=m["SSIM"],
                        time_s=float(info["time_s"])))
        print(f"  converged lam={lam} -> PSNR {m['PSNR']:.4f} "
              f"({info['n_iter']} it, {info['time_s']:.1f}s)")
    with open(os.path.join(OUT, "penalty_invariance.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(inv[0].keys()))
        w.writeheader()
        w.writerows(inv)

    # Adopted setting: the PSNR-optimal alphas, symmetric penalties (see report S5.4 -
    # the fixed point is lambda-independent, so the asymmetric stage-3 winner is an
    # artefact of early stopping; symmetric 2.0 is numerically better conditioned).
    final = dict(alpha0=float(best1["alpha0"]), alpha1=float(best1["alpha1"]),
                 beta=float(best1["beta"]), lam1=2.0, lam2=2.0, lam3=2.0,
                 tuning_variance=VAR, tuning_metric="PSNR",
                 psnr_stage1=float(best1["PSNR"]), ssim_stage1=float(best1["SSIM"]),
                 stage3_best=dict(lam1=float(best3["lam1"]), lam2=float(best3["lam2"]),
                                  lam3=float(best3["lam3"]), psnr=float(best3["PSNR"])),
                 noisy_psnr=float(base["PSNR"]), noisy_ssim=float(base["SSIM"]))
    with open(os.path.join(OUT, "chosen_parameters.json"), "w") as fh:
        json.dump(final, fh, indent=2)
    print("\nFINAL:", json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
