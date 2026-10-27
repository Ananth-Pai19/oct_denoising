"""Shared plotting style for all figures (publication-quality defaults)."""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "results", "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "font.size": 9.5,
    "axes.titlesize": 10,
    "axes.labelsize": 9.5,
    "axes.grid": False,
    "legend.fontsize": 8.5,
    "legend.frameon": False,
    "image.cmap": "gray",
    "image.interpolation": "nearest",
    "lines.linewidth": 1.4,
})

__all__ = ["plt", "FIG", "ROOT", "save", "show_gray", "panel_labels"]


def save(fig, name, subdir=""):
    d = os.path.join(FIG, subdir) if subdir else FIG
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, name)
    fig.savefig(path)
    print(f"  figure -> {os.path.relpath(path, ROOT)}")
    return path


def show_gray(ax, img, title=None, vmin=0.0, vmax=1.0, **kw):
    im = ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, **kw)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, pad=3)
    return im


def panel_labels(fig, axes, labels=None, dx=-0.02, dy=1.03):
    """(a), (b), ... labels above each panel."""
    import string
    labels = labels or list(string.ascii_lowercase)
    for ax, lab in zip(np.ravel(axes), labels):
        ax.text(dx, dy, f"({lab})", transform=ax.transAxes,
                fontsize=9, va="bottom", ha="right")


import numpy as np  # noqa: E402  (kept at the bottom to avoid a circular-looking import order)
