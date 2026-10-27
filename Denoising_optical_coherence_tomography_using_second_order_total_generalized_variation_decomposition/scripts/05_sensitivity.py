"""
Step 3.4 - parameter sensitivity.

Four 1-D sweeps of the TGV^2 model on the synthetic phantom, each at three noise levels,
plus a 2-D (alpha1, beta) heat map:

    alpha0            (alpha1 = beta = 0.1 fixed)
    alpha1            (alpha0 = beta = 0.1 fixed)
    beta              (alpha0 = alpha1 = 0.1 fixed)
    alpha0/alpha1 ratio (alpha1 = 0.1, balanced-geometric mean kept at 0.1)
    lambda1=lambda2=lambda3  (alpha0 = alpha1 = beta = 0.1)

Outputs
    results/sensitivity/sweeps.csv
    results/sensitivity/heatmap.csv
    results/figures/sensitivity/fig_sensitivity.png
    results/figures/sensitivity/fig_heatmap.png
    results/figures/sensitivity/fig_beta_visual.png
"""

from __future__ import annotations

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.metrics import all_metrics                                  # noqa: E402
from src.phantom import add_gaussian_noise, synthetic_oct_phantom    # noqa: E402
from src.plots import plt, save, show_gray                           # noqa: E402
from src.tgv import tgv_decompose                                    # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "sensitivity")
os.makedirs(OUT, exist_ok=True)

BASE = dict(alpha0=0.1, alpha1=0.1, beta=0.1, lam1=2.0, lam2=2.0, lam3=2.0)
VARS = [0.02, 0.06, 0.10]
GRIDS = {
    "alpha0": [0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
    "alpha1": [0.02, 0.05, 0.1, 0.2, 0.5],
    "beta": [0.005, 0.02, 0.05, 0.1, 0.2, 0.4],
    "ratio": [0.25, 0.5, 1.0, 2.0, 4.0],
    "lambda": [0.5, 1.0, 2.0, 5.0, 10.0],
}


def run(f, **kw):
    p = dict(BASE)
    p.update(kw)
    return tgv_decompose(f, p["alpha0"], p["alpha1"], p["beta"],
                         p["lam1"], p["lam2"], p["lam3"], max_iter=400, tol=1e-3)


def config(name, value):
    kw = {}
    if name == "alpha0":
        kw["alpha0"] = value
    elif name == "alpha1":
        kw["alpha1"] = value
    elif name == "beta":
        kw["beta"] = value
    elif name == "ratio":
        kw["alpha0"] = BASE["alpha1"] * value
    elif name == "lambda":
        kw.update(lam1=value, lam2=value, lam3=value)
    return kw


def main():
    clean = synthetic_oct_phantom()
    rows = []
    for var in VARS:
        f = add_gaussian_noise(clean, var, seed=1)
        for name, grid in GRIDS.items():
            for v in grid:
                u, _ = run(f, **config(name, v))
                m = all_metrics(clean, u)
                rows.append(dict(param=name, value=v, var=var, PSNR=m["PSNR"],
                                 SSIM=m["SSIM"], RMSE=m["RMSE"]))
                print(f"var={var} {name}={v:<6} PSNR {m['PSNR']:.2f} SSIM {m['SSIM']:.4f}")
    with open(os.path.join(OUT, "sweeps.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---------------- 1-D figures ---------------------------------------------------
    fig, axes = plt.subplots(2, 5, figsize=(14, 5.0))
    for j, name in enumerate(GRIDS):
        for i, metric in enumerate(["PSNR", "SSIM"]):
            ax = axes[i, j]
            for var in VARS:
                sel = [r for r in rows if r["param"] == name and r["var"] == var]
                ax.plot([r["value"] for r in sel], [r[metric] for r in sel],
                        marker="o", ms=3.5, label=f"var={var}")
            ax.set_xscale("log")
            ax.set_xlabel(name.replace("lambda", r"$\lambda_1=\lambda_2=\lambda_3$")
                          .replace("ratio", r"$\alpha_0/\alpha_1$"))
            ax.set_ylabel(metric)
            ax.set_title(f"{metric} vs {name}")
            if i == 0 and j == 0:
                ax.legend()
    fig.tight_layout()
    save(fig, "fig_sensitivity.png", "sensitivity")
    plt.close(fig)

    # ---------------- 2-D heat map ---------------------------------------------------
    var = 0.06
    f = add_gaussian_noise(clean, var, seed=1)
    a1s = [0.03, 0.05, 0.08, 0.12, 0.2, 0.3]
    bs = [0.01, 0.03, 0.06, 0.1, 0.2, 0.35]
    P = np.zeros((len(a1s), len(bs)))
    S = np.zeros_like(P)
    for i, a1 in enumerate(a1s):
        for j, b in enumerate(bs):
            u, _ = tgv_decompose(f, a1, a1, b, 2.0, 2.0, 2.0, max_iter=400, tol=1e-3)
            m = all_metrics(clean, u)
            P[i, j], S[i, j] = m["PSNR"], m["SSIM"]
    with open(os.path.join(OUT, "heatmap.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["alpha1", "beta", "PSNR", "SSIM"])
        for i, a1 in enumerate(a1s):
            for j, b in enumerate(bs):
                w.writerow([a1, b, P[i, j], S[i, j]])

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for ax, (M, lab) in zip(axes, [(P, "PSNR (dB)"), (S, "SSIM")]):
        im = ax.imshow(M, origin="lower", aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(bs)))
        ax.set_xticklabels([f"{b:g}" for b in bs])
        ax.set_yticks(range(len(a1s)))
        ax.set_yticklabels([f"{a:g}" for a in a1s])
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel(r"$\alpha_1=\alpha_0$")
        ax.set_title(lab)
        fig.colorbar(im, ax=ax)
        # mark the chosen value
        i0 = int(np.argmin(np.abs(np.array(a1s) - 0.1)))
        j0 = int(np.argmin(np.abs(np.array(bs) - 0.1)))
        ax.plot(j0, i0, "rx", ms=10, mew=2)
    fig.tight_layout()
    save(fig, "fig_heatmap.png", "sensitivity")
    plt.close(fig)

    # ---------------- visual effect of beta -----------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
    show_gray(axes[0], f, f"noisy (var={var})")
    for ax, b in zip(axes[1:], [0.02, 0.1, 0.4]):
        u, _ = tgv_decompose(f, 0.1, 0.1, b, 2.0, 2.0, 2.0, max_iter=400, tol=1e-3)
        m = all_metrics(clean, u)
        show_gray(ax, np.clip(u, 0, 1),
                  f"beta={b}  PSNR {m['PSNR']:.1f} dB")
    fig.tight_layout()
    save(fig, "fig_beta_visual.png", "sensitivity")
    plt.close(fig)
    print("done")


if __name__ == "__main__":
    main()
