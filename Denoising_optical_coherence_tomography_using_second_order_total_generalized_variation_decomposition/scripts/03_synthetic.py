"""
Step 3.1 - synthetic experiments: reproduction of the paper's Table 2, Fig. 3 and Fig. 4.

Runs all methods on the 9-layer phantom at noise variances {0.02,0.04,0.06,0.08,0.10} and
writes

    results/synthetic/table2.csv          long-format metrics (method x variance x metric)
    results/synthetic/table2.md           Table-2-style markdown table
    results/synthetic/table2_paper.md     the same table with the paper's numbers alongside
    results/synthetic/timings.csv         wall-clock per method
    results/figures/synthetic/fig3_metrics.png
    results/figures/synthetic/fig4_comparison.png
    results/figures/synthetic/fig4_zoom.png
    results/figures/synthetic/fig_decomposition.png
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.baselines import METHODS                                    # noqa: E402
from src.metrics import all_metrics                                 # noqa: E402
from src.phantom import add_gaussian_noise, synthetic_oct_phantom   # noqa: E402
from src.plots import plt, save, show_gray                          # noqa: E402
from src.tgv import tgv_decompose, vo_decompose                     # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "synthetic")
os.makedirs(OUT, exist_ok=True)

# tuned baseline settings (scripts/02b_baseline_tuning.py); empty dicts if not tuned yet
try:
    with open(os.path.join(ROOT, "results", "baselines", "best_params.json")) as _fh:
        BEST = json.load(_fh)
except FileNotFoundError:                                       # pragma: no cover
    BEST = {}


def method_callables():
    """name -> callable(img) using the tuned parameters."""
    import functools
    out = {}
    for name, fn in METHODS.items():
        cfg = BEST.get(name, {})
        if name == "HWT":
            # the sweep selected the universal threshold with c = 1.0; keep it *adaptive*
            # (sigma estimated per image) instead of freezing one absolute number
            cfg = {"levels": cfg.get("levels", 3)}
        out[name] = functools.partial(fn, **cfg)
    return out

VARIANCES = [0.02, 0.04, 0.06, 0.08, 0.10]
P = dict(alpha0=0.1, alpha1=0.1, beta=0.1, lam1=2.0, lam2=2.0, lam3=2.0)
SEED = 1

# numbers transcribed from Table 2 of Duan et al. (2016), for the comparison table only
PAPER = {
    "Noisy": {"PSNR": [17.5325, 14.9554, 13.5430, 12.5612, 11.8256],
              "SNR": [8.4672, 5.8901, 4.7777, 3.4959, 2.7603],
              "RMSE": [9.9006, 10.3206, 10.5214, 10.6182, 10.6609],
              "SSIM": [0.1327, 0.0978, 0.0826, 0.0730, 0.0655]},
    "HMF": {"PSNR": [21.0741, 20.2947, 19.9165, 19.4774, 19.0066],
            "SNR": [12.0088, 11.2294, 10.8512, 10.4121, 9.9413],
            "RMSE": [6.0263, 6.5034, 6.8127, 7.0316, 7.0406],
            "SSIM": [0.6844, 0.5745, 0.4979, 0.4496, 0.4133]},
    "HWT": {"PSNR": [24.3981, 22.1582, 20.8430, 19.3209, 18.0995],
            "SNR": [15.3328, 13.0929, 11.7777, 10.2556, 9.0342],
            "RMSE": [4.8105, 4.9782, 5.0147, 5.0320, 5.0487],
            "SSIM": [0.9249, 0.8923, 0.8709, 0.8369, 0.8273]},
    "DTCWT": {"PSNR": [27.8702, 24.7867, 23.1115, 21.9070, 20.4468],
              "SNR": [18.8049, 15.7214, 14.0462, 12.8417, 11.3815],
              "RMSE": [4.1782, 4.4722, 4.5374, 4.5669, 4.5834],
              "SSIM": [0.9317, 0.9094, 0.8934, 0.8730, 0.8513]},
    "NCDF": {"PSNR": [20.9252, 19.0848, 18.4835, 17.3998, 16.8213],
             "SNR": [11.8599, 10.0195, 9.4182, 8.3345, 7.7560],
             "RMSE": [4.7767, 4.8449, 4.8746, 4.9408, 5.0166],
             "SSIM": [0.7998, 0.7808, 0.7781, 0.7768, 0.7722]},
    "ACED": {"PSNR": [28.1206, 26.2074, 24.6428, 23.0592, 21.4316],
             "SNR": [19.0553, 17.1421, 15.5775, 13.9939, 12.3664],
             "RMSE": [5.0821, 5.4730, 5.4769, 5.5201, 5.5256],
             "SSIM": [0.7679, 0.6761, 0.6016, 0.5141, 0.4654]},
    "VO": {"PSNR": [28.2919, 26.4631, 24.9032, 23.5448, 22.3510],
           "SNR": [19.2266, 17.3978, 15.8379, 14.4795, 13.2857],
           "RMSE": [4.0315, 4.0812, 4.0943, 4.0973, 4.1099],
           "SSIM": [0.9728, 0.9624, 0.9517, 0.9394, 0.9267]},
    "TGV": {"PSNR": [32.0284, 29.5787, 27.4901, 25.9198, 24.4470],
            "SNR": [22.9631, 20.5134, 18.4248, 16.8545, 15.3817],
            "RMSE": [3.7756, 3.9375, 3.9437, 3.9456, 3.9759],
            "SSIM": [0.9871, 0.9781, 0.9684, 0.9590, 0.9481]},
}
ORDER = ["Noisy", "HMF", "HWT", "DTCWT", "NCDF", "ACED", "VO", "TGV"]


def run_all():
    clean = synthetic_oct_phantom()
    methods = method_callables()
    rows, timings, images = [], {}, {}
    for var in VARIANCES:
        noisy = add_gaussian_noise(clean, var, seed=SEED)
        images[var] = {"Noisy": noisy, "Clean": clean}
        m = all_metrics(clean, noisy)
        rows.append(dict(var=var, method="Noisy", **m, time_s=0.0))

        for name, fn in methods.items():
            t0 = time.perf_counter()
            out = fn(noisy)
            dt = time.perf_counter() - t0
            images[var][name] = out
            m = all_metrics(clean, out)
            rows.append(dict(var=var, method=name, **m, time_s=dt))
            timings.setdefault(name, []).append(dt)

        t0 = time.perf_counter()
        u_vo, _ = vo_decompose(noisy, alpha1=P["alpha1"], beta=P["beta"],
                               lam1=P["lam1"], lam2=P["lam2"], max_iter=300, tol=1e-3)
        dt = time.perf_counter() - t0
        images[var]["VO"] = u_vo
        rows.append(dict(var=var, method="VO", **all_metrics(clean, u_vo), time_s=dt))
        timings.setdefault("VO", []).append(dt)

        t0 = time.perf_counter()
        u_tgv, tex, info = tgv_decompose(noisy, P["alpha0"], P["alpha1"], P["beta"],
                                         P["lam1"], P["lam2"], P["lam3"],
                                         max_iter=300, tol=1e-3, return_info=True)
        dt = time.perf_counter() - t0
        images[var]["TGV"] = u_tgv
        images[var]["texture"] = tex
        rows.append(dict(var=var, method="TGV", **all_metrics(clean, u_tgv), time_s=dt,
                         n_iter=info["n_iter"]))
        timings.setdefault("TGV", []).append(dt)
        print(f"var={var}: TGV PSNR {rows[-1]['PSNR']:.2f} dB, "
              f"{info['n_iter']} it, {dt:.2f} s")
    return rows, timings, images, clean


def write_tables(rows):
    fields = ["var", "method", "PSNR", "SNR", "RMSE", "RMSE_255", "SSIM", "time_s",
              "n_iter"]
    with open(os.path.join(OUT, "table2.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # markdown, Table-2 layout: one block per metric
    lookup = {(r["method"], r["var"]): r for r in rows}
    md = ["# Table 2 (reproduction): synthetic 9-layer phantom", "",
          "Metrics on the [0,1] scale; PSNR peak = 1.0; SSIM = skimage default (7x7 uniform).",
          "Parameters (all noise levels, all methods) are the frozen ones from "
          "`results/tuning/chosen_parameters.json`.", ""]
    for metric in ["PSNR", "SNR", "RMSE", "SSIM"]:
        md.append(f"## {metric}")
        md.append("")
        md.append("| Method | " + " | ".join(f"var={v}" for v in VARIANCES) + " |")
        md.append("|---|" + "---|" * len(VARIANCES))
        for meth in ORDER:
            cells = []
            for v in VARIANCES:
                r = lookup.get((meth, v))
                cells.append(f"{r[metric]:.4f}" if r else "-")
            md.append(f"| {meth} | " + " | ".join(cells) + " |")
        md.append("")
    open(os.path.join(OUT, "table2.md"), "w").write("\n".join(md))

    md = ["# Reproduced vs. published Table 2", "",
          "`ours / paper` for PSNR (dB) and SSIM.", "",
          "| Method | " + " | ".join(f"var={v}" for v in VARIANCES) + " |",
          "|---|" + "---|" * len(VARIANCES)]
    for meth in ORDER:
        cells = []
        for i, v in enumerate(VARIANCES):
            r = lookup.get((meth, v))
            if r is None:
                cells.append("-")
            else:
                cells.append(f"{r['PSNR']:.2f} / {PAPER[meth]['PSNR'][i]:.2f}")
        md.append(f"| {meth} (PSNR) | " + " | ".join(cells) + " |")
    md.append("")
    md.append("| Method | " + " | ".join(f"var={v}" for v in VARIANCES) + " |")
    md.append("|---|" + "---|" * len(VARIANCES))
    for meth in ORDER:
        cells = []
        for i, v in enumerate(VARIANCES):
            r = lookup.get((meth, v))
            if r is None:
                cells.append("-")
            else:
                cells.append(f"{r['SSIM']:.3f} / {PAPER[meth]['SSIM'][i]:.3f}")
        md.append(f"| {meth} (SSIM) | " + " | ".join(cells) + " |")
    open(os.path.join(OUT, "table2_paper.md"), "w").write("\n".join(md))


def fig_metrics(rows):
    lookup = {(r["method"], r["var"]): r for r in rows}
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 2.9))
    specs = [("PSNR", "PSNR (dB)"), ("SNR", "SNR (dB)"),
             ("RMSE", "RMSE"), ("SSIM", "SSIM")]
    styles = {"Noisy": ("0.6", "o"), "HMF": ("C7", "s"), "HWT": ("C8", "^"),
              "DTCWT": ("C2", "v"), "NCDF": ("C1", "D"), "ACED": ("C4", "P"),
              "VO": ("C3", "x"), "TGV": ("C0", "*")}
    for ax, (metric, label) in zip(axes, specs):
        for meth in ORDER:
            ys = [lookup.get((meth, v), {}).get(metric, np.nan) for v in VARIANCES]
            c, mk = styles[meth]
            ax.plot(VARIANCES, ys, marker=mk, color=c, ms=4.5 if meth != "TGV" else 8,
                    lw=2.0 if meth == "TGV" else 1.2, label=meth)
        ax.set_xlabel(r"noise variance $\sigma^2$")
        ax.set_ylabel(label)
        ax.set_title(label)
    axes[0].legend(ncol=2, fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig3_metrics.png", "synthetic")
    plt.close(fig)


def fig_comparison(images, var=0.02):
    clean = images[var]["Clean"]
    order = ["Noisy", "HMF", "HWT", "DTCWT", "NCDF", "ACED", "VO", "TGV"]
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.6))
    for ax, meth in zip(axes.ravel(), order):
        img = images[var][meth]
        p = all_metrics(clean, img)
        show_gray(ax, np.clip(img, 0, 1), f"{meth}  PSNR {p['PSNR']:.2f} dB")
    fig.tight_layout()
    save(fig, "fig4_comparison.png", "synthetic")
    plt.close(fig)


def fig_zoom(images, var=0.02):
    clean = images[var]["Clean"]
    m, n = clean.shape
    r0, r1 = int(0.25 * m), int(0.25 * m) + 84
    c0, c1 = int(0.30 * n), int(0.30 * n) + 110
    order = ["Noisy", "HMF", "HWT", "DTCWT", "NCDF", "ACED", "VO", "TGV"]
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.2))
    for ax, meth in zip(axes.ravel(), order):
        show_gray(ax, np.clip(images[var][meth], 0, 1)[r0:r1, c0:c1], meth)
    fig.tight_layout()
    save(fig, "fig4_zoom.png", "synthetic")
    plt.close(fig)


def fig_decomposition(images, var=0.06):
    clean = images[var]["Clean"]
    noisy = images[var]["Noisy"]
    tex = images[var]["texture"]
    u = images[var]["TGV"]
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.0))
    show_gray(axes[0], noisy, f"input f (var={var})")
    show_gray(axes[1], u, "structure u (TGV$^2$)")
    show_gray(axes[2], tex, "texture f-u (= div g)", vmin=-0.4, vmax=0.4)
    ax = axes[3]
    ax.hist(tex.ravel(), bins=120, color="0.3")
    ax.set_yscale("log")
    ax.set_title("texture histogram")
    ax.set_xlabel("f - u")
    fig.tight_layout()
    save(fig, "fig_decomposition.png", "synthetic")
    plt.close(fig)


def main():
    rows, timings, images, clean = run_all()
    write_tables(rows)
    with open(os.path.join(OUT, "timings.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "mean_time_s", "per_variance_s"])
        for meth, ts in timings.items():
            w.writerow([meth, f"{np.mean(ts):.3f}", ";".join(f"{t:.3f}" for t in ts)])
    fig_metrics(rows)
    fig_comparison(images)
    fig_zoom(images)
    fig_decomposition(images)
    np.save(os.path.join(OUT, "phantom_clean.npy"), clean)
    print("done")


if __name__ == "__main__":
    main()
