"""Image-quality metrics used in the paper's Table 2 / Fig. 3.

Definitions (stated explicitly because the paper does not state them):

    MSE   = mean((u - u_hat)^2)          over all pixels, images in [0, 1]
    RMSE  = sqrt(MSE)
    PSNR  = 10 log10(peak^2 / MSE)       peak = 1.0 (data range), see psnr(peak=...)
    SNR   = 10 log10(||u||^2 / ||u - u_hat||^2)
    SSIM  = mean SSIM map, Wang et al. 2004, skimage defaults (7x7 uniform window)

Note: the paper's RMSE column is not consistent with its own PSNR/SNR columns (see
report/paper_summary.md, item 4), so RMSE is also returned both on the [0,1] scale and on
the 0-255 scale for easier comparison with other work.
"""

from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity

__all__ = ["mse", "rmse", "psnr", "snr", "ssim", "all_metrics"]


def mse(clean, test):
    clean = np.asarray(clean, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    return float(np.mean((clean - test) ** 2))


def rmse(clean, test, scale=1.0):
    return float(scale * np.sqrt(mse(clean, test)))


def psnr(clean, test, peak=1.0):
    m = mse(clean, test)
    if m <= 0:
        return float("inf")
    return float(10.0 * np.log10(peak ** 2 / m))


def snr(clean, test):
    clean = np.asarray(clean, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    num = float(np.sum(clean ** 2))
    den = float(np.sum((clean - test) ** 2))
    if den <= 0:
        return float("inf")
    return float(10.0 * np.log10(num / den))


def ssim(clean, test, data_range=1.0, gaussian_weights=False):
    return float(structural_similarity(np.asarray(clean, dtype=np.float64),
                                       np.asarray(test, dtype=np.float64),
                                       data_range=data_range,
                                       gaussian_weights=gaussian_weights))


def all_metrics(clean, test, peak=1.0, gaussian_weights=False):
    """PSNR / SNR / RMSE (both scales) / SSIM in one dict."""
    return {
        "PSNR": psnr(clean, test, peak=peak),
        "SNR": snr(clean, test),
        "RMSE": rmse(clean, test, 1.0),
        "RMSE_255": rmse(clean, test, 255.0),
        "SSIM": ssim(clean, test, gaussian_weights=gaussian_weights),
    }
