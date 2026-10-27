"""
Step 2 (verification) - operator/closed-form unit tests, convergence diagnostics and the
piecewise-affine (staircase) test.

Outputs
    results/verification/operator_tests.txt
    results/figures/convergence/fig_convergence.png
    results/figures/convergence/fig_staircase.png
    results/verification/convergence_stats.json
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import verify                                              # noqa: E402
from src.metrics import all_metrics                                 # noqa: E402
from src.phantom import add_gaussian_noise, piecewise_affine_image, synthetic_oct_phantom  # noqa: E402
from src.plots import plt, save, show_gray                          # noqa: E402
from src.tgv import tgv_decompose, vo_decompose                     # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "verification")
os.makedirs(OUT, exist_ok=True)

# frozen parameters from scripts/02_tune_parameters.py
P = dict(alpha0=0.1, alpha1=0.1, beta=0.1, lam1=2.0, lam2=2.0, lam3=2.0)


def convergence_curve():
    clean = synthetic_oct_phantom()
    noisy = add_gaussian_noise(clean, 0.04, seed=1)
    u, tex, h = tgv_decompose(noisy, P["alpha0"], P["alpha1"], P["beta"],
                              P["lam1"], P["lam2"], P["lam3"],
                              max_iter=400, tol=1e-6, return_info=True)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))
    ax = axes[0]
    ax.semilogy(h["rel_change"], "k")
    ax.axhline(1e-3, color="r", ls="--", lw=1)
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$\|u^k-u^{k-1}\|_2/\|u^k\|_2$")
    ax.set_title("relative change of u (red: 1e-3 stopping rule)")
    ax.set_xlim(0, min(len(h["rel_change"]), 200))

    ax = axes[1]
    ax.plot(h["energy_model"], "k", label=r"model energy (2.11)")
    ax.set_xlabel("iteration")
    ax.set_ylabel("energy")
    ax.set_title("objective of the model")
    ax.set_xlim(0, min(len(h["energy_model"]), 200))

    ax = axes[2]
    ax.plot(h["data_term"], label="data term")
    ax.plot(h["tv_g"], label=r"$\beta\|g\|_1$")
    ax.plot(h["tv_w"], label=r"$\alpha_1\|\nabla u-p\|_1$")
    ax.plot(h["tv_v"], label=r"$\alpha_0\|\epsilon p\|_1$")
    ax.set_xlabel("iteration")
    ax.set_ylabel("term value")
    ax.set_title("individual terms of (2.11)")
    ax.set_xlim(0, min(len(h["data_term"]), 200))
    ax.legend(loc="best")

    fig.tight_layout()
    save(fig, "fig_convergence.png", "convergence")
    plt.close(fig)

    stats = dict(n_iter_converged=int(h["n_iter"]),
                 time_s=float(h["time_s"]),
                 energy_model_final=float(h["energy_model"][-1]),
                 energy_model_first=float(h["energy_model"][0]),
                 rel_change_final=float(h["rel_change"][-1]),
                 metrics=all_metrics(clean, u))
    # a shorter run with the paper's stopping rule, for the runtime table
    u2, _, h2 = tgv_decompose(noisy, P["alpha0"], P["alpha1"], P["beta"],
                              P["lam1"], P["lam2"], P["lam3"],
                              max_iter=300, tol=1e-3, return_info=True)
    stats["n_iter_tol1e-3"] = int(h2["n_iter"])
    stats["time_s_tol1e-3"] = float(h2["time_s"])
    stats["metrics_tol1e-3"] = all_metrics(clean, u2)
    with open(os.path.join(OUT, "convergence_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)
    print(json.dumps(stats, indent=2))
    return stats


def staircase_test():
    clean = piecewise_affine_image(shape=(192, 256))
    noisy = add_gaussian_noise(clean, 0.03, seed=2)

    # Both models are tuned by PSNR on this test image (a controlled synthetic demo, so
    # tuning here is legitimate and is reported).  TV: 1-D sweep over alpha1.
    tv_best = None
    for a1 in (0.05, 0.08, 0.10, 0.15):
        u, _ = vo_decompose(noisy, alpha1=a1, beta=0.10, lam1=2.0, lam2=2.0,
                            max_iter=500, tol=1e-6)
        p = all_metrics(clean, u)["PSNR"]
        if tv_best is None or p > tv_best[0]:
            tv_best = (p, a1, u)
    print(f"  TV best: alpha1={tv_best[1]} PSNR={tv_best[0]:.2f}")
    u_tv = tv_best[2]

    tgv_best = None
    for a0, a1, b in [(0.1, 0.1, 0.1), (0.3, 0.1, 0.1), (0.4, 0.1, 0.1),
                      (0.5, 0.1, 0.1), (0.4, 0.12, 0.1), (0.4, 0.08, 0.1)]:
        u, _ = tgv_decompose(noisy, a0, a1, b, 2.0, 2.0, 2.0, max_iter=500, tol=1e-6)
        p = all_metrics(clean, u)["PSNR"]
        if tgv_best is None or p > tgv_best[0]:
            tgv_best = (p, (a0, a1, b), u)
    print(f"  TGV best: (a0,a1,beta)={tgv_best[1]} PSNR={tgv_best[0]:.2f}")
    u_tgv = tgv_best[2]

    row = clean.shape[0] // 2
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6))
    show_gray(axes[0, 0], clean, "clean (piecewise affine)")
    show_gray(axes[0, 1], noisy, r"noisy, $\sigma^2$=0.03")
    show_gray(axes[0, 2], np.abs(clean - u_tv), "|error|, TV", vmin=0, vmax=0.12)
    ax = axes[1, 0]
    ax.plot(clean[row], "k", lw=2, label="clean")
    ax.plot(noisy[row], color="0.75", lw=0.6, label="noisy")
    ax.set_title(f"intensity profile, row {row}")
    ax.legend(loc="upper right")
    ax = axes[1, 1]
    ax.plot(clean[row], "k", lw=3, alpha=0.5, label="clean")
    ax.plot(u_tv[row], "C3", lw=1.4, label="TV (VO)")
    ax.set_title("TV: staircases in the affine ramps")
    ax.legend(loc="upper right")
    ax = axes[1, 2]
    ax.plot(clean[row], "k", lw=3, alpha=0.5, label="clean")
    ax.plot(u_tgv[row], "C0", lw=1.4, label="TGV$^2$")
    ax.set_title("TGV$^2$: ramps preserved")
    ax.legend(loc="upper right")
    fig.tight_layout()
    save(fig, "fig_staircase.png", "convergence")
    plt.close(fig)

    # zoom on one ramp for the paper-style figure
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(clean[row], "k", lw=2.6, alpha=0.55, label="clean")
    ax.plot(u_tv[row], "C3", lw=1.5, label="TV (VO)")
    ax.plot(u_tgv[row], "C0", lw=1.5, label="TGV$^2$")
    ax.set_xlim(8, 60)
    ax.set_xlabel("pixel (column)")
    ax.set_ylabel("intensity")
    ax.legend()
    ax.set_title("staircase effect: TV vs TGV$^2$ (zoom on one ramp)")
    fig.tight_layout()
    save(fig, "fig_staircase_zoom.png", "convergence")
    plt.close(fig)

    # ---- quantitative staircase measures -------------------------------------------
    # inside each ramp, the true first difference is constant (slope s).  A staircase
    # shows up as many pixels where the local difference is ~0.
    def staircase_stats(u, clean_row):
        diffs_c = np.diff(clean_row)
        out = []
        for lo, hi in [(4, 60), (68, 124), (132, 188), (196, 252)]:
            s = np.median(diffs_c[lo:hi])
            d = np.diff(u[lo:hi])
            out.append(dict(slope=float(s),
                            flat_fraction=float(np.mean(np.abs(d) < 0.25 * abs(s))),
                            slope_ratio=float(np.polyfit(np.arange(hi - lo + 1),
                                                         u[lo:hi + 1], 1)[0] / s)))
        return out

    st_tv = staircase_stats(u_tv[row], clean[row])
    st_tgv = staircase_stats(u_tgv[row], clean[row])
    res = {
        "tv_alpha1": tv_best[1],
        "tgv_params": dict(zip(("alpha0", "alpha1", "beta"), tgv_best[1])),
        "metrics_tv": all_metrics(clean, u_tv),
        "metrics_tgv": all_metrics(clean, u_tgv),
        "metrics_noisy": all_metrics(clean, noisy),
        "staircase_TV": st_tv,
        "staircase_TGV": st_tgv,
        "flat_fraction_TV": float(np.mean([s["flat_fraction"] for s in st_tv])),
        "flat_fraction_TGV": float(np.mean([s["flat_fraction"] for s in st_tgv])),
        "slope_ratio_TV": float(np.mean([s["slope_ratio"] for s in st_tv])),
        "slope_ratio_TGV": float(np.mean([s["slope_ratio"] for s in st_tgv])),
    }
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("staircase_")},
                     indent=2))
    with open(os.path.join(OUT, "staircase_stats.json"), "w") as fh:
        json.dump(res, fh, indent=2)


def phantom_figure():
    clean = synthetic_oct_phantom()
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.0))
    show_gray(axes[0], clean, "9-layer phantom (Eq. 4.1 analogue)")
    ax = axes[1]
    ax.plot(clean[:, clean.shape[1] // 2], "k")
    ax.invert_yaxis()
    ax.set_xlabel("intensity")
    ax.set_ylabel("depth (row)")
    ax.set_title("central cross-section (Fig. 2 middle)")
    ax = axes[2]
    ax.plot(clean[clean.shape[0] // 2], "k")
    ax.set_xlabel("lateral (column)")
    ax.set_ylabel("intensity")
    ax.set_title("central row")
    fig.tight_layout()
    save(fig, "fig_phantom.png", "synthetic")
    plt.close(fig)


def tgv_tv_limit():
    """alpha0 -> infinity: the TGV^2 term degenerates to TV, so TGV must reproduce VO."""
    clean = synthetic_oct_phantom()
    f = add_gaussian_noise(clean, 0.04, seed=1)
    uv, _ = vo_decompose(f, alpha1=0.1, beta=0.1, lam1=2.0, lam2=2.0,
                         max_iter=500, tol=1e-6)
    ut, _ = tgv_decompose(f, alpha0=100.0, alpha1=0.1, beta=0.1,
                          lam1=2.0, lam2=2.0, lam3=2.0, max_iter=500, tol=1e-6)
    res = {"max_abs_diff": float(np.abs(ut - uv).max()),
           "psnr_tv": all_metrics(clean, uv)["PSNR"],
           "psnr_tgv_large_alpha0": all_metrics(clean, ut)["PSNR"]}
    print(json.dumps(res, indent=2))
    with open(os.path.join(OUT, "tgv_tv_limit.json"), "w") as fh:
        json.dump(res, fh, indent=2)


def main():
    print("== operator / closed-form verification ==")
    ok = verify.report()
    with open(os.path.join(OUT, "operator_tests.txt"), "w") as fh:
        fh.write("adjoint consistency + FFT closed form vs direct sparse solve\n")
        for name, fn in (("adjoints", verify.check_adjoints),
                         ("closed_forms", verify.check_closed_forms)):
            for k, v in fn().items():
                fh.write(f"{name}.{k}: {v:.6e}\n")
        fh.write(f"RESULT: {'PASS' if ok else 'FAIL'}\n")
    print("\n== convergence diagnostics ==")
    convergence_curve()
    print("\n== staircase test (TV vs TGV) ==")
    staircase_test()
    print("\n== TGV -> TV limit ==")
    tgv_tv_limit()
    print("\n== phantom figure ==")
    phantom_figure()


if __name__ == "__main__":
    main()
