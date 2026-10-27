"""
Synthetic OCT phantom with 9 retinal layers, built in the spirit of Eq. (4.1) of
Duan et al. (2016):

    f(x, y) = sum_i (a_i/2) exp( -((x-x_i)^2 + (y-y_i)^2) / (2 sigma_i^2) ) - b_i

The paper does **not** report a_i, b_i, sigma_i or the layer centres (only "by choosing
appropriate parameters"), so this is a documented re-construction, not a bit-exact copy of
their Fig. 2.  Each layer is a Gaussian profile in depth centred on a *curved* interface
y_i(x) (foveal dip + slight tilt), which is the same image class: piecewise-smooth layers
with sharp boundaries, hyper-reflective ILM/RPE, hypo-reflective GCL/INL/ONL.

The nine layers, top to bottom, follow the paper's caption:
    ILM, NFL, GCL, IPL, INL, OPL, ONL, IS, OS (+ RPE as the 9th reflective band).

Default amplitudes were chosen so that the phantom has mean(u^2) ~ 0.12 and max = 1,
i.e. the same dynamic range implied by the paper's noisy-image PSNR of 17.5 dB at
variance 0.02 (see report/paper_summary.md item 3).
"""

from __future__ import annotations

import numpy as np

__all__ = ["LAYERS", "synthetic_oct_phantom", "add_gaussian_noise", "cross_section"]

#: (name, reflectivity a_i, depth y_i (fraction of height), Gaussian sigma_i in pixels)
#: depths are fractions of the image height; sigma in pixels at the default 256-row size.
LAYERS = [
    ("ILM/NFL", 1.00, 0.30, 2.2),
    ("NFL/GCL", 0.55, 0.36, 1.6),
    ("GCL",     0.34, 0.42, 3.4),
    ("IPL",     0.62, 0.49, 3.0),
    ("INL",     0.30, 0.56, 4.0),
    ("OPL",     0.55, 0.62, 1.5),
    ("ONL",     0.22, 0.68, 4.5),
    ("IS/OS",   0.85, 0.75, 1.3),
    ("RPE",     0.95, 0.80, 2.6),
]


def synthetic_oct_phantom(shape=(256, 256), layers=None, dip=0.055, tilt=0.012,
                          ripple=0.004, background=0.02, width_scale=1.35, seed=0):
    """Build a clean 9-layer OCT B-scan in [0, 1].

    Parameters
    ----------
    shape : (M, N)
        Output size (rows = depth, columns = lateral).
    layers : list of (name, amplitude, depth_fraction, sigma_pixels) or None
        Layer table; defaults to :data:`LAYERS`.
    dip : float
        Depth of the foveal pit (parabolic, as a fraction of the image height).
    tilt : float
        Linear tilt of the retinal surface across the B-scan.
    ripple : float
        Amplitude of a small sinusoidal undulation of the interfaces.
    background : float
        Constant offset added to the whole image (vitreous haze).
    width_scale : float
        Multiplies every layer's sigma (calibrated so that mean(u^2) ~ 0.09, i.e. a noisy
        SNR within ~2 dB of the value implied by the paper's Table 2 at variance 0.02).
    seed : int
        Seed for the (very small) lateral jitter of the layer centres.
    """
    layers = LAYERS if layers is None else layers
    m, n = shape
    rng = np.random.default_rng(seed)

    y = np.arange(m, dtype=np.float64)[:, None]          # (m, 1)
    x = np.linspace(-1.0, 1.0, n, dtype=np.float64)[None, :]   # (1, n)

    # lateral shape of every interface: parabola (foveal dip) + tilt + smooth ripple
    curve = (dip * (x ** 2) + tilt * x
             + ripple * np.sin(3.0 * np.pi * x) + ripple * 0.5 * np.cos(7.0 * np.pi * x))

    img = np.full((m, n), float(background), dtype=np.float64)
    scale = m / 256.0
    for k, (name, amp, depth, sigma) in enumerate(layers):
        jitter = 0.004 * rng.standard_normal()
        y0 = (depth + jitter) * m
        yc = y0 + m * curve                                # (1, n) curved interface
        s = sigma * scale * width_scale
        band = amp * np.exp(-((y - yc) ** 2) / (2.0 * s ** 2))
        img += band

    # weak inter-layer haze so that the vitreous is not exactly zero (as in real B-scans)
    img += 0.05 * np.exp(-((y - m * 0.55) / (m * 0.16)) ** 2)
    img = np.maximum(img, 0.0)
    img /= img.max()
    return np.clip(img, 0.0, 1.0)


def add_gaussian_noise(img, variance, seed=0):
    """Add zero-mean i.i.d. Gaussian noise of the given variance (paper's protocol)."""
    rng = np.random.default_rng(seed)
    return np.asarray(img, dtype=np.float64) + rng.normal(0.0, np.sqrt(variance), img.shape)


def cross_section(img):
    """Central column of the phantom (Fig. 2, middle panel)."""
    return np.asarray(img)[:, img.shape[1] // 2]


def piecewise_affine_image(shape=(128, 128)):
    """Piecewise-affine test image for the staircase experiment.

    Four vertical bands, each an exact linear ramp in x with a different slope, separated
    by sharp jumps; plus a small global linear ramp in y so the image is genuinely 2-D.
    TGV^2 can represent this image exactly (it is piecewise affine), TV cannot (it is
    piecewise constant), which is precisely the effect the paper attributes to TGV.
    """
    m, n = shape
    x = np.linspace(0.0, 1.0, n)[None, :]
    y = np.linspace(0.0, 1.0, m)[:, None]
    edges = [0.0, 0.25, 0.5, 0.75, 1.0]
    base = [0.12, 0.62, 0.28, 0.78]
    slope = [0.30, -0.28, 0.22, 0.10]
    u = np.zeros((m, n))
    for k in range(4):
        lo, hi = edges[k], edges[k + 1]
        sel = (x >= lo) & (x < hi)
        u = np.where(sel, base[k] + slope[k] * (x - lo) / (hi - lo), u)
    u = u + 0.12 * y
    return np.clip(u, 0.0, 1.0)
