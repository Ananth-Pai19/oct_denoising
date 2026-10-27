"""
Comparison methods from Table 1 of Duan et al. (2016).

Implemented here (no deep learning, no training):

    hmf    hybrid median filter (Nieminen et al. 1987)
    hwt    Haar wavelet, hard thresholding (universal/MAD threshold)
    dtcwt  dual-tree complex wavelet, soft thresholding (dtcwt package)
    ncdf   nonlinear complex diffusion filter (Gilboa et al. 2004; Salinas & Cabrera 2007)
    aced   anisotropic coherence-enhancing diffusion (Weickert 1998) - optional, see note

Every method takes and returns an image in the same [0,1] (log) domain as the TGV solver,
so the metrics are directly comparable.  None of these methods has a "variance" input;
the paper reports no parameter values for any of them, so the settings below are standard
choices (documented per function) and are tuned *once* on a single noise level
(scripts/03_baselines_tuning.py) rather than per experiment.

ACED is included but is the least reliable of the set: its structure-tensor / coherence
diffusion needs a time step, an integration time and a contrast parameter, none of which
are reported, and its result is very sensitive to them.  Its numbers should be read as
"a reasonable ACED", not "their ACED".
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

# dtcwt 0.13 (the version on PyPI) predates NumPy 2.0 and uses two removed aliases.
if not hasattr(np, "asfarray"):          # pragma: no cover - compatibility shim
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)  # noqa: E731
if not hasattr(np, "issubsctype"):       # pragma: no cover
    np.issubsctype = lambda a, b: np.issubdtype(a, b)  # noqa: E731

__all__ = ["hmf", "hwt", "dtcwt_denoise", "ncdf", "aced", "estimate_sigma", "METHODS"]

# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def estimate_sigma(img, patch=8):
    """Robust noise estimate: median absolute deviation of the diagonal detail band.

    Uses the classic Immerkaer/Donoho estimator on a Haar diagonal sub-band, which is
    insensitive to edges.
    """
    a = np.asarray(img, dtype=np.float64)
    m, n = (a.shape[0] // 2) * 2, (a.shape[1] // 2) * 2
    a = a[:m, :n]
    # diagonal Haar detail
    d = (a[0::2, 0::2] - a[0::2, 1::2] - a[1::2, 0::2] + a[1::2, 1::2]) / 2.0
    return float(np.median(np.abs(d - np.median(d))) / 0.6745)


# --------------------------------------------------------------------------------------
# 1. hybrid median filter
# --------------------------------------------------------------------------------------
def _shift_stack(a, offsets):
    return np.stack([np.roll(np.roll(a, dy, axis=0), dx, axis=1) for dy, dx in offsets])


def hmf(img, iterations=1):
    """Hybrid median filter (5x5): median of {3x3 cross median, 3x3 diagonal median, centre}.

    This is the standard HMF of Nieminen, Heinonen & Neuvo (1987); it preserves thin lines
    better than a plain median because the two sub-masks are oriented.
    """
    a = np.asarray(img, dtype=np.float64).copy()
    cross = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0), (0, -2), (0, 2), (-2, 0), (2, 0)]
    diag = [(0, 0), (-1, -1), (-1, 1), (1, -1), (1, 1), (-2, -2), (-2, 2), (2, -2), (2, 2)]
    for _ in range(iterations):
        mc = np.median(_shift_stack(a, cross), axis=0)
        md = np.median(_shift_stack(a, diag), axis=0)
        out = np.median(np.stack([mc, md, a]), axis=0)
        a = out
    return a


# --------------------------------------------------------------------------------------
# 2. Haar wavelet hard thresholding
# --------------------------------------------------------------------------------------
def hwt(img, levels=3, wavelet="haar", sigma=None, threshold=None, mode="hard"):
    """Wavelet thresholding.  Default threshold = universal sigma*sqrt(2 ln N)."""
    import pywt

    a = np.asarray(img, dtype=np.float64)
    if sigma is None:
        sigma = estimate_sigma(a)
    if threshold is None:
        threshold = sigma * np.sqrt(2.0 * np.log(a.size))
    coeffs = pywt.wavedec2(a, wavelet=wavelet, level=levels, mode="periodization")
    new = [coeffs[0]]
    for detail in coeffs[1:]:
        if mode == "hard":
            new.append(tuple(pywt.threshold(d, threshold, mode="hard") for d in detail))
        else:
            new.append(tuple(pywt.threshold(d, threshold, mode="soft") for d in detail))
    return pywt.waverec2(new, wavelet=wavelet, mode="periodization")[:a.shape[0], :a.shape[1]]


# --------------------------------------------------------------------------------------
# 3. dual-tree complex wavelet soft thresholding
# --------------------------------------------------------------------------------------
def dtcwt_denoise(img, levels=4, sigma=None, k=3.0, biort="near_sym_a", qshift="qshift_a"):
    """Dual-tree CWT soft thresholding with per-level noise scaling.

    The DT-CWT of white noise has a noise gain of ~sqrt(2/3) per level relative to the
    input std, so a fixed threshold in every sub-band would over-smooth coarse levels;
    the threshold is therefore scaled by the empirical std of each sub-band (MAD), which
    is the standard adaptive rule and does not need the noise level as an input.
    """
    from dtcwt.numpy import Transform2d

    a = np.asarray(img, dtype=np.float64)
    tr = Transform2d(biort=biort, qshift=qshift)
    pyr = tr.forward(a, nlevels=levels)
    out_high = []
    for hi in pyr.highpasses:
        # hi: (rows, cols, 6) complex coefficients; the real part is Gaussian with the
        # sub-band noise std, so MAD of Re(c) is the correct robust scale estimate.
        re = np.real(hi)
        s = np.median(np.abs(re - np.median(re))) / 0.6745
        thr = k * s
        mag = np.abs(hi)
        scale = np.maximum(mag - thr, 0.0) / np.maximum(mag, 1e-12)
        out_high.append(hi * scale)
    pyr.highpasses = tuple(out_high)
    rec = tr.inverse(pyr)
    return np.real(rec)[:a.shape[0], :a.shape[1]]


# --------------------------------------------------------------------------------------
# 4. nonlinear complex diffusion filter
# --------------------------------------------------------------------------------------
def ncdf(img, theta=np.pi / 30.0, k=0.25, t=0.6, dt=None, n_iter=None):
    """Nonlinear complex diffusion filter (Gilboa, Sochen & Zeevi 2004).

        I_t = div( c(Im I) grad I ),   c = exp(i*theta) / (1 + (Im(I)/(k*theta))^2)

    The imaginary part acts as an edge detector (it is approximately theta*Delta I for
    small theta), so diffusion is strong in flat areas and weak at edges, while the real
    part is what is kept.  Explicit Euler with the standard 5-point Laplacian and a
    stability-limited step dt <= 0.24.

    Parameters follow the common OCT setting (Salinas & Cabrera Fernandez 2007):
    theta = pi/30, k = 0.25 (image in [0,1]), total diffusion time t = 0.6.
    """
    a = np.asarray(img, dtype=np.float64)
    if dt is None:
        dt = 0.2
    if n_iter is None:
        n_iter = max(1, int(round(t / dt)))
    dt = min(dt, 0.24)

    I = a.astype(np.complex128)
    for _ in range(n_iter):
        im = np.imag(I)
        c = np.exp(1j * theta) / (1.0 + (im / (k * theta)) ** 2)
        lap = (np.roll(I, 1, 0) + np.roll(I, -1, 0)
               + np.roll(I, 1, 1) + np.roll(I, -1, 1) - 4.0 * I)
        # c is complex and spatially varying: use the (real) magnitude of c per pixel
        I = I + dt * (c * lap)
    return np.real(I)


# --------------------------------------------------------------------------------------
# 5. anisotropic coherence-enhancing diffusion (optional)
# --------------------------------------------------------------------------------------
def aced(img, sigma=1.0, rho=6.0, t=1.5, dt=None, n_iter=None, alpha=0.001, C=1e-6):
    """Coherence-enhancing anisotropic diffusion (Weickert 1998, eq. 6.x).

    Diffusion tensor from the smoothed structure tensor with eigenvalues
    lambda1 = alpha, lambda2 = 1 if mu1-mu2 > C else alpha + (1-alpha)*exp(-C/(mu1-mu2)^2),
    which enhances line-like (layer) structures.  Explicit Euler, dt limited by 0.24.

    NOTE: this is a faithful standard implementation, but the paper reports none of
    (sigma, rho, t, C), so its numbers are indicative only.
    """
    a = np.asarray(img, dtype=np.float64)
    if dt is None:
        dt = 0.15
    if n_iter is None:
        n_iter = max(1, int(round(t / dt)))
    dt = min(dt, 0.24)

    I = a.copy()
    for _ in range(n_iter):
        # pre-smooth the gradient (sigma), then smooth the tensor (rho)
        gy, gx = np.gradient(ndimage.gaussian_filter(I, sigma))
        jxx = ndimage.gaussian_filter(gx * gx, rho)
        jyy = ndimage.gaussian_filter(gy * gy, rho)
        jxy = ndimage.gaussian_filter(gx * gy, rho)
        # eigen-decomposition of the 2x2 symmetric structure tensor
        tmp = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy ** 2)
        mu1 = 0.5 * (jxx + jyy + tmp)
        mu2 = 0.5 * (jxx + jyy - tmp)
        diff = np.maximum(mu1 - mu2, 0.0)
        lam2 = np.where(diff > C, 1.0, alpha + (1 - alpha) * np.exp(-C / (diff + 1e-12) ** 2))
        lam1 = np.full_like(lam2, alpha)
        # eigenvector of mu1: (cos t, sin t) with t = 0.5*atan2(2*jxy, jxx-jyy)
        theta_ = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
        ct, st = np.cos(theta_), np.sin(theta_)
        # D = lam1 * v1 v1^T + lam2 * v2 v2^T, v1 = (ct, st), v2 = (-st, ct)
        d11 = lam1 * ct * ct + lam2 * st * st
        d22 = lam1 * st * st + lam2 * ct * ct
        d12 = (lam1 - lam2) * ct * st
        # divergence of D grad I (forward/backward differences)
        ix = np.roll(I, -1, 1) - I
        iy = np.roll(I, -1, 0) - I
        fx = d11 * ix + d12 * iy
        fy = d12 * ix + d22 * iy
        div = (fx - np.roll(fx, 1, 1)) + (fy - np.roll(fy, 1, 0))
        I = I + dt * div
    return I


#: name -> callable, used by the experiment scripts
METHODS = {
    "HMF": hmf,
    "HWT": hwt,
    "DTCWT": dtcwt_denoise,
    "NCDF": ncdf,
    "ACED": aced,
}
