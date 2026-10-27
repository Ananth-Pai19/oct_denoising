"""
Step 0 - inspect the data: what is in Sparsity_SDOCT_DATASET_2012, and is it already
log-compressed?

Writes results/inspection/
    dataset_summary.txt     sizes, dtypes, intensity statistics
    noise_model.txt         additive-vs-multiplicative test (sigma_local ~ mu_local^p)
    fig_dataset_overview.png  three raw frames + their averaged counterparts
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.io_utils import (DATASET_DIR, list_cases, load_tif, noise_model_slope,  # noqa: E402
                          to_unit)
from src.plots import FIG, plt, save, show_gray                      # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "inspection")
os.makedirs(OUT, exist_ok=True)


def main():
    cases = list_cases(DATASET_DIR)
    lines = [f"Sparsity_SDOCT_DATASET_2012: {len(cases)} cases", ""]
    for cid, raw, avg in cases:
        a = load_tif(raw)
        b = load_tif(avg) if avg else None
        lines.append(f"case {cid:>2}  raw {a.shape} {a.dtype} range {a.min():.0f}-{a.max():.0f}"
                     f"  mean {a.mean():.1f} std {a.std():.1f}"
                     + (f" | avg {b.shape} range {b.min():.0f}-{b.max():.0f} "
                        f"mean {b.mean():.1f} std {b.std():.1f}" if b is not None else ""))
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(OUT, "dataset_summary.txt"), "w").write(txt + "\n")

    # ---- additive vs multiplicative: sigma_local as a function of mu_local -------------
    nlines = ["Local-noise test:  log(sigma_local) = p * log(mu_local) + c",
              "p ~ 0  -> additive noise (log-domain data)",
              "p ~ 1  -> multiplicative noise (linear-domain data)", ""]
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.9))
    for ax, (cid, raw, avg) in zip(axes, cases[:3]):
        a = to_unit(load_tif(raw))
        p_raw, bm, bs = noise_model_slope(a)
        p_avg = None
        if avg:
            b = to_unit(load_tif(avg))
            p_avg, bm2, bs2 = noise_model_slope(b)
            ax.loglog(bm2, bs2, "o-", ms=3, lw=1, label=f"averaged (p={p_avg:.2f})")
        ax.loglog(bm, bs, "s-", ms=3, lw=1, label=f"raw (p={p_raw:.2f})")
        ax.set_xlabel(r"local mean $\mu$")
        ax.set_ylabel(r"local std $\sigma$")
        ax.set_title(f"case {cid}")
        ax.legend()
        nlines.append(f"case {cid:>2}: p_raw = {p_raw:.3f}"
                      + (f", p_averaged = {p_avg:.3f}" if p_avg is not None else ""))
    save(fig, "noise_model_slope.png", "inspection")
    plt.close(fig)

    # also test the *linearised* hypothesis: log(1+x) is what we would apply if the data
    # were linear; if the data were linear, the p of the log-transformed version would
    # drop towards 0.  We report both, so the choice is evidence-based.
    nlines.append("")
    for cid, raw, avg in cases[:3]:
        a = to_unit(load_tif(raw))
        p0, _, _ = noise_model_slope(a)
        p1, _, _ = noise_model_slope(np.log1p(50.0 * a))
        nlines.append(f"case {cid:>2}: p(no transform) = {p0:.3f}, "
                      f"p(log(1+50x)) = {p1:.3f}")
    txt = "\n".join(nlines)
    print(txt)
    open(os.path.join(OUT, "noise_model.txt"), "w").write(txt + "\n")

    # ---- overview figure -------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.4))
    for j, (cid, raw, avg) in enumerate(cases[:3]):
        show_gray(axes[0, j], to_unit(load_tif(raw)), f"case {cid}: raw")
        if avg:
            show_gray(axes[1, j], to_unit(load_tif(avg)), f"case {cid}: averaged")
    fig.tight_layout()
    save(fig, "fig_dataset_overview.png", "inspection")
    plt.close(fig)


if __name__ == "__main__":
    main()
