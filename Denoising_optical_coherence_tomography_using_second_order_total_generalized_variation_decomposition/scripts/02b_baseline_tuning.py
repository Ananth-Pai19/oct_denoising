"""
Baseline tuning: every comparison method has at least one free parameter that the paper
does not report.  Tuning only the proposed method would be unfair, so each baseline is
tuned *once*, on the synthetic phantom at variance 0.04 (the same image and noise level
used to tune TGV^2), by PSNR.  The winning settings are written to
results/baselines/best_params.json and consumed by scripts/03_synthetic.py and
scripts/04_real_data.py.

    results/baselines/baseline_grid.csv
    results/baselines/best_params.json
"""

from __future__ import annotations

import csv
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import baselines as B                                    # noqa: E402
from src.metrics import all_metrics                               # noqa: E402
from src.phantom import add_gaussian_noise, synthetic_oct_phantom  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "baselines")
os.makedirs(OUT, exist_ok=True)
VAR = 0.04


def main():
    clean = synthetic_oct_phantom()
    noisy = add_gaussian_noise(clean, VAR, seed=1)

    grids = {
        "HMF": [dict(iterations=k) for k in (1, 2, 3)],
        "HWT": [dict(levels=lv, threshold=c * B.estimate_sigma(noisy)
                     * np.sqrt(2 * np.log(noisy.size)))
                for lv in (2, 3, 4) for c in (0.5, 1.0, 1.5, 2.0)],
        "DTCWT": [dict(levels=lv, k=k) for lv in (3, 4) for k in (1.0, 2.0, 3.0, 4.0, 5.0)],
        "NCDF": [dict(k=k, t=t) for k in (0.1, 0.25, 0.5) for t in (0.3, 0.6, 1.0)],
        "ACED": [dict(t=t, rho=r) for t in (1.0, 2.0) for r in (2.0, 4.0, 8.0)],
    }

    rows = []
    best = {}
    for name, cfgs in grids.items():
        fn = B.METHODS[name]
        scores = []
        for cfg in cfgs:
            out = fn(noisy, **cfg)
            m = all_metrics(clean, out)
            scores.append((m["PSNR"], m["SSIM"], cfg, m))
            rows.append(dict(method=name, params=json.dumps(cfg),
                             PSNR=m["PSNR"], SSIM=m["SSIM"]))
        scores.sort(key=lambda s: (-s[0], -s[1]))
        psnr, ssim, cfg, m = scores[0]
        best[name] = cfg
        print(f"{name:6s} best {cfg} -> PSNR {psnr:.2f} SSIM {ssim:.4f}")
        for s in scores[:3]:
            print(f"        {s[2]} {s[0]:.2f}")

    with open(os.path.join(OUT, "baseline_grid.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(OUT, "best_params.json"), "w") as fh:
        json.dump(best, fh, indent=2)
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
