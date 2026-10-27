"""
Loading and conditioning of the OCT data in ``Sparsity_SDOCT_DATASET_2012``.

What the data actually are (checked in ``scripts/00_inspect_data.py``):

    * 8-bit TIFF, 508x940 ... 562x1000, uint8, values 0-255
    * "<n>_Raw Image.tif"      single-frame (noisy) B-scan  -> used as test data
    * "<n>_Averaged Image.tif" multi-frame average          -> used as a visual reference
                                                              ("less noisy", not ground truth)
    * "S_MSBTD result.tif"     280-row crop of the MSBTD dictionary-denoised result of
                               Fang et al. (2012) - a third-party reference, not a GT

The values are **already log-compressed**: the dynamic range is 8-bit with a non-zero
mid-grey pedestal in the vitreous and a unimodal distribution with a long right tail on
the near-noise-free averages (skew 1.0-1.4), which is the signature of a post-log display
mapping.  Linear |signal|^2 OCT data would instead be Rayleigh-distributed, near-zero in
the vitreous, and strongly right-skewed even after averaging.
``scripts/00_inspect_data.py`` also reports the local-std vs local-mean slope, which is
flat (slope ~ 0.0-0.2 in log-log) for additive noise and ~1 for multiplicative noise.

Therefore: no additional log transform is applied to this dataset, only x/255.  If a
linear-intensity file is supplied, pass ``log_domain=False`` to apply log(1 + x) first.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import tifffile
from scipy import ndimage

__all__ = ["DATASET_DIR", "list_cases", "load_tif", "to_unit", "to_log",
           "load_case", "noise_model_slope"]


DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "Sparsity_SDOCT_DATASET_2012")


def load_tif(path):
    """Read a TIFF as float64 (raw 0-255 values)."""
    return tifffile.imread(path).astype(np.float64)


def to_unit(a, lo=None, hi=255.0):
    """Scale to [0, 1]."""
    a = np.asarray(a, dtype=np.float64)
    lo = 0.0 if lo is None else lo
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


def to_log(a, eps=1.0):
    """log(1 + x) compression for *linear-intensity* input, renormalised to [0, 1]."""
    a = np.asarray(a, dtype=np.float64)
    a = a - a.min()
    out = np.log1p(a / max(eps, 1e-12))
    mx = out.max()
    return out / mx if mx > 0 else out


def list_cases(dataset_dir=DATASET_DIR):
    """Sorted list of (case_id, raw_path, averaged_path) for the 17 datasets."""
    cases = []
    for d in sorted(glob.glob(os.path.join(dataset_dir, "*")),
                    key=lambda p: (not os.path.basename(p).isdigit(),
                                   int(os.path.basename(p)) if os.path.basename(p).isdigit() else 0)):
        if not os.path.isdir(d):
            continue
        cid = os.path.basename(d)
        raws = glob.glob(os.path.join(d, "*Raw Image.tif"))
        avgs = glob.glob(os.path.join(d, "*Averaged Image.tif"))
        if raws:
            cases.append((cid, raws[0], avgs[0] if avgs else None))
    return cases


def load_case(raw_path, log_domain=True, hi=255.0):
    """Load one B-scan as a float image in [0,1] in the (log) domain used by the model."""
    a = load_tif(raw_path)
    if not log_domain:
        return to_log(a)
    return to_unit(a, hi=hi)


def noise_model_slope(img, win=7, nbins=12):
    """Empirical exponent p in  sigma_local ~ mu_local^p.

    p ~ 0    -> additive noise (log domain)
    p ~ 1    -> multiplicative noise (linear domain)

    Estimated by regressing log(sigma_local) on log(mu_local) over a grid of box filters,
    using only blocks whose local mean is well above the pedestal.
    """
    a = np.asarray(img, dtype=np.float64)
    mu = ndimage.uniform_filter(a, win)
    var = ndimage.uniform_filter(a * a, win) - mu ** 2
    sd = np.sqrt(np.maximum(var, 1e-12))
    m = mu.ravel()
    s = sd.ravel()
    lo, hi = np.percentile(m, 5), np.percentile(m, 99)
    sel = (m > lo) & (m < hi) & (s > 0)
    m, s = m[sel], s[sel]
    bins = np.linspace(m.min(), m.max(), nbins + 1)
    idx = np.clip(np.digitize(m, bins) - 1, 0, nbins - 1)
    bm = np.array([m[idx == k].mean() for k in range(nbins) if np.any(idx == k)])
    bs = np.array([s[idx == k].mean() for k in range(nbins) if np.any(idx == k)])
    ok = (bm > 0) & (bs > 0)
    p = np.polyfit(np.log(bm[ok]), np.log(bs[ok]), 1)[0]
    return float(p), bm, bs
