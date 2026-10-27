"""
Step 3.2 - real data: run every method on B-scans from Sparsity_SDOCT_DATASET_2012.

There is no noise-free ground truth for these frames, so in addition to figures we report
three *reference-free* proxies (all computed on the [0,1] log-domain image):

  sigma_flat   std of the residual inside automatically selected flat blocks (the 10% of
               16x16 blocks with the smallest local std in the input).  Lower = more
               noise removed *in homogeneous regions*; it says nothing about details.
  ENL          equivalent number of looks, mean^2/var over the same flat blocks.
  ssim_vs_avg  SSIM against the multi-frame averaged image of the same eye, used as a
               pseudo-reference.  It penalises both residual noise and structural
               distortion, but the averaged image is not ground truth (it is itself
               smoothed and is a different realisation), so it is only a *relative* guide.

Outputs
  results/real/table_real.csv
  results/figures/real/fig_case{id}_compare.png
  results/figures/real/fig_case{id}_zoom.png
  results/figures/real/fig_vs_averaged.png
"""

from __future__ import annotations

import csv
import functools
import json
import os
import sys
import time

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.baselines import METHODS                                     # noqa: E402
from src.io_utils import DATASET_DIR, list_cases, load_case           # noqa: E402
from src.metrics import ssim                                          # noqa: E402
from src.plots import plt, save, show_gray                            # noqa: E402
from src.tgv import tgv_decompose, vo_decompose                       # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "real")
os.makedirs(OUT, exist_ok=True)

CASES = [1, 9, 13]          # three frames with different contrast / layer visibility
P = dict(alpha0=0.1, alpha1=0.1, beta=0.1, lam1=2.0, lam2=2.0, lam3=2.0)
ORDER = ["Noisy", "HMF", "HWT", "DTCWT", "NCDF", "ACED", "VO", "TGV"]

try:
    with open(os.path.join(ROOT, "results", "baselines", "best_params.json")) as _fh:
        BEST = json.load(_fh)
except FileNotFoundError:                                            # pragma: no cover
    BEST = {}


def method_callables():
    out = {}
    for name, fn in METHODS.items():
        cfg = BEST.get(name, {})
        if name == "HWT":
            cfg = {"levels": cfg.get("levels", 3)}
        out[name] = functools.partial(fn, **cfg)
    return out


def flat_stats(img, ref_img, block=16, frac=0.10):
    """Noise level inside automatically-selected homogeneous blocks of ``ref_img``."""
    m, n = (img.shape[0] // block) * block, (img.shape[1] // block) * block
    a = img[:m, :n]
    sd = np.zeros((m // block, n // block))
    mean = np.zeros_like(sd)
    r = ref_img[:m, :n]
    for i in range(sd.shape[0]):
        for j in range(sd.shape[1]):
            blk = r[i * block:(i + 1) * block, j * block:(j + 1) * block]
            sd[i, j] = blk.std()
            mean[i, j] = blk.mean()
    thr = np.quantile(sd, frac)
    sel = sd <= thr
    vals = np.concatenate([a[i * block:(i + 1) * block, j * block:(j + 1) * block].ravel()
                           for i, j in zip(*np.where(sel))])
    mu = np.concatenate([mean[sel].ravel()]).mean()
    sigma = float(vals.std())
    return dict(sigma_flat=sigma,
                ENL=float(mu ** 2 / max(sigma ** 2, 1e-12)),
                flat_blocks=int(sel.sum()))


def plateau_fraction(img, ref, thresh=1e-3, min_level=0.2):
    """Fraction of neighbouring pixel pairs inside the tissue that are exactly flat.

    A staircased (piecewise-constant) estimate produces long runs of identical values, so
    this fraction is high for TV and low for TGV.  Only pairs inside the retina
    (input local mean above ``min_level``) are counted, so the black vitreous does not
    dominate the statistic.
    """
    a = np.clip(np.asarray(img, dtype=np.float64), 0, 1)
    mu = ndimage.uniform_filter(np.asarray(ref, dtype=np.float64), 9)
    tissue = mu > min_level
    dx = np.abs(np.diff(a, axis=1))
    dy = np.abs(np.diff(a, axis=0))
    tx = tissue[:, :-1] & tissue[:, 1:]
    ty = tissue[:-1, :] & tissue[1:, :]
    flat = np.concatenate([dx[tx] < thresh, dy[ty] < thresh])
    return float(np.mean(flat))


def main():
    cases = {int(c[0]): c for c in list_cases(DATASET_DIR)}
    methods = method_callables()
    rows = []

    for cid in CASES:
        _, raw_path, avg_path = cases[cid]
        f = load_case(raw_path)
        avg = load_case(avg_path) if avg_path else None
        print(f"case {cid}: {f.shape}")

        outs = {"Noisy": f}
        times = {}
        for name, fn in methods.items():
            t0 = time.perf_counter()
            outs[name] = fn(f)
            times[name] = time.perf_counter() - t0
        t0 = time.perf_counter()
        outs["VO"], _ = vo_decompose(f, alpha1=P["alpha1"], beta=P["beta"],
                                     lam1=P["lam1"], lam2=P["lam2"],
                                     max_iter=300, tol=1e-3)
        times["VO"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        outs["TGV"], tex, info = tgv_decompose(f, P["alpha0"], P["alpha1"], P["beta"],
                                               P["lam1"], P["lam2"], P["lam3"],
                                               max_iter=300, tol=1e-3, return_info=True)
        times["TGV"] = time.perf_counter() - t0
        print(f"  TGV: {info['n_iter']} iterations, {times['TGV']:.1f} s")

        for name in ORDER:
            st = flat_stats(outs[name], f)
            row = dict(case=cid, method=name, time_s=times.get(name, 0.0), **st,
                       plateau=plateau_fraction(outs[name], f))
            if avg is not None:
                row["ssim_vs_avg"] = ssim(avg, np.clip(outs[name], 0, 1))
            rows.append(row)
            print(f"  {name:6s} sigma_flat {st['sigma_flat']:.4f} "
                  f"ENL {st['ENL']:7.2f} plateaus {row['plateau']*100:5.1f}%  "
                  f"t={times.get(name, 0):.2f}s")

        # ---- figure: 2x4 comparison ------------------------------------------------
        fig, axes = plt.subplots(2, 4, figsize=(13, 6.2))
        for ax, name in zip(axes.ravel(), ORDER):
            show_gray(ax, np.clip(outs[name], 0, 1), name)
        fig.tight_layout()
        save(fig, f"fig_case{cid}_compare.png", "real")
        plt.close(fig)

        # ---- figure: zoom on the layer boundaries ----------------------------------
        m, n = f.shape
        r0, r1 = int(0.30 * m), int(0.30 * m) + 120
        c0, c1 = int(0.35 * n), int(0.35 * n) + 150
        fig, axes = plt.subplots(2, 5, figsize=(13, 5.4))
        for ax, name in zip(axes.ravel(), ORDER + ["Averaged"]):
            img = avg if name == "Averaged" else outs[name]
            show_gray(ax, np.clip(img, 0, 1)[r0:r1, c0:c1], name)
        fig.tight_layout()
        save(fig, f"fig_case{cid}_zoom.png", "real")
        plt.close(fig)

        # ---- strength-matched TGV (see report S5.3): at equal alpha, TGV's deformation
        # field absorbs part of the gradient, so it is *gentler* than VO.  Scale its
        # parameters until the residual noise in flat blocks matches VO's, then compare
        # edge sharpness.  The scale factor is searched, not tuned per image by hand.
        target = next(r for r in rows if r["case"] == cid and r["method"] == "VO")["sigma_flat"]
        best = None
        for scale in (1.5, 2.0, 3.0, 4.0):
            u_s, _ = tgv_decompose(f, P["alpha0"] * scale, P["alpha1"] * scale,
                                   P["beta"], P["lam1"], P["lam2"], P["lam3"],
                                   max_iter=300, tol=1e-3)
            st = flat_stats(u_s, f)
            if best is None or abs(st["sigma_flat"] - target) < abs(best[1] - target):
                best = (scale, st["sigma_flat"], u_s)
        scale, sig, u_match = best
        rows.append(dict(case=cid, method=f"TGV_matched(x{scale:g})", time_s=0.0,
                         **flat_stats(u_match, f),
                         plateau=plateau_fraction(u_match, f),
                         ssim_vs_avg=ssim(avg, np.clip(u_match, 0, 1)) if avg is not None
                         else np.nan))
        print(f"  TGV strength-matched: scale {scale} -> sigma_flat {sig:.4f} "
              f"(VO {target:.4f})")
        fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
        for ax, (nm, im) in zip(axes, [("VO", outs["VO"]),
                                       (f"TGV x1", outs["TGV"]),
                                       (f"TGV x{scale:g} (matched)", u_match),
                                       ("Averaged", avg)]):
            show_gray(ax, np.clip(im, 0, 1)[r0:r1, c0:c1], nm)
        fig.tight_layout()
        save(fig, f"fig_case{cid}_matched.png", "real")
        plt.close(fig)

    with open(os.path.join(OUT, "table_real.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # ---- reference-free summary figure ----------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.1))
    for ax, (key, label) in zip(axes, [("sigma_flat", r"residual $\sigma$ in flat blocks"),
                                       ("ENL", "ENL in flat blocks (higher = smoother)"),
                                       ("ssim_vs_avg", "SSIM vs. averaged image")]):
        w = 0.8 / len(CASES)
        for k, cid in enumerate(CASES):
            vals = [next(r for r in rows if r["case"] == cid and r["method"] == nm)[key]
                    for nm in ORDER]
            ax.bar(np.arange(len(ORDER)) + k * w, vals, width=w, label=f"case {cid}")
        ax.set_xticks(np.arange(len(ORDER)) + w * (len(CASES) - 1) / 2)
        ax.set_xticklabels(ORDER, rotation=45, ha="right")
        ax.set_title(label)
    axes[0].legend()
    fig.tight_layout()
    save(fig, "fig_real_summary.png", "real")
    plt.close(fig)


if __name__ == "__main__":
    main()
