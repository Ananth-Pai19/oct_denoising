"""
The four OCT denoisers of this repository behind one interface, for comparison.ipynb.

Every denoiser takes a raw B-scan on the 8-bit grey scale (float, 0-255) and returns the
denoised image on the same scale, so all of them can be scored by the same metric code.

    tgv_denoise   Duan et al. (2016), TGV^2 decomposition      -> imported from ../Denoising_..._decomposition/src
    nlm_denoise   Chen & Gao (2022), FCFDS-kernel NLM          -> ported from ../NLM_PSO_Denoising/NLM_Denoising.ipynb
    wavelet_mst   Subhedar et al. (2023), modified soft
                  threshold wavelet (+ PSO for its parameters) -> ported from ../NLM_PSO_Denoising/PSO_Denoising.ipynb
    ksvd_denoise  Zhang et al. (2022), K-SVD global dictionary
                  + OMP with GLCM/Laplacian noise estimation   -> ported from ../OMP_KSVD/denoising.ipynb

The ports are copies of the notebook code with the module-level constants turned into
keyword arguments, so the strength parameters can be tuned from the comparison notebook.
Nothing about the algorithms is changed, with one exception: the wavelet port skips the
paper's log(1 + z) step by default (log_domain=True), because the Duke B-scans are already
log-compressed. Taking a second log costs about 5.6 dB (see comparison.ipynb, section 1).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pywt
from numpy.lib.stride_tricks import sliding_window_view
from scipy import ndimage
from scipy.signal import convolve2d
from skimage.feature import graycomatrix
from skimage.metrics import structural_similarity

REPO = Path(__file__).resolve().parents[1]
TGV_DIR = REPO / "Denoising_optical_coherence_tomography_using_second_order_total_generalized_variation_decomposition"
KSVD_CHECKPOINT = REPO / "OMP_KSVD" / "outputs" / "checkpoints" / "global_dict.npz"
PEAK = 255.0

sys.path.insert(0, str(TGV_DIR))
from src.tgv import tgv_decompose  # noqa: E402


# ======================================================================================
# Metrics — one definition for every method
# ======================================================================================
def psnr(ref, img, peak=PEAK):
    m = np.mean((np.asarray(ref, np.float64) - np.asarray(img, np.float64)) ** 2)
    return np.inf if m == 0 else float(10 * np.log10(peak ** 2 / m))


def ssim(ref, img, peak=PEAK):
    # Wang et al. (2004): 11x11 Gaussian window, sigma 1.5, fixed 8-bit data range
    return float(structural_similarity(np.asarray(ref, np.float64), np.asarray(img, np.float64),
                                       data_range=peak, gaussian_weights=True, sigma=1.5,
                                       use_sample_covariance=False))


def _sobel_mag(img):
    x = np.asarray(img, np.float64)
    return np.hypot(ndimage.sobel(x, axis=1, mode="reflect"), ndimage.sobel(x, axis=0, mode="reflect"))


def epi(ref, img):
    # Edge Preservation Index: correlation coefficient of the Sobel edge maps of reference and test
    a, b = _sobel_mag(ref), _sobel_mag(img)
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else 0.0


def all_metrics(ref, img):
    return {"PSNR": psnr(ref, img), "SSIM": ssim(ref, img), "EPI": epi(ref, img)}


# ======================================================================================
# 1. TGV^2 decomposition (Duan et al. 2016)
# ======================================================================================
TGV_PAPER = dict(alpha0=0.1, alpha1=0.1, beta=0.1, lam1=2.0, lam2=2.0, lam3=2.0)


def tgv_denoise(img, alpha=0.1, beta=0.1, lam=2.0, max_iter=300, tol=1e-3):
    # The solver works on [0, 1]; the dataset is already log-compressed (see TGV src/io_utils.py)
    u, _ = tgv_decompose(np.asarray(img, np.float64) / PEAK, alpha0=alpha, alpha1=alpha, beta=beta,
                         lam1=lam, lam2=lam, lam3=lam, max_iter=max_iter, tol=tol)
    return np.clip(u * PEAK, 0, PEAK)


# ======================================================================================
# 2. Non-local means with a fractional compact finite-difference kernel (Chen & Gao 2022)
# ======================================================================================
NLM_SEARCH, NLM_PATCH = 21, 7
NLM_ALPHA, NLM_TAPS, NLM_PHI_X = 0.5, 4, 4.0
_OFFS = np.arange(-(NLM_PATCH // 2), NLM_PATCH // 2 + 1)


def fcfds_rho(alpha):
    return ((24 + 17 * alpha + 3 * alpha ** 2) / 24,
            -(11 * alpha + 3 * alpha ** 2) / 12,
            (5 * alpha + 3 * alpha ** 2) / 24)


def fcfds_coeffs(alpha=NLM_ALPHA, n_taps=NLM_TAPS):
    g = np.ones(n_taps)
    for j in range(1, n_taps):
        g[j] = g[j - 1] * (j - 1 - alpha) / j
    rho1, rho2, rho3 = fcfds_rho(alpha)
    c = rho1 * g
    c[1:] += rho2 * g[:-1]
    c[2:] += rho3 * g[:-2]
    return c


def frac_diff(f, c, axis):
    # D[i] = sum_j c[j] * f[i - j] along `axis`, with replicated borders
    n = len(c)
    pad = [(0, 0), (0, 0)]
    pad[axis] = (n - 1, 0)
    fp = np.pad(f, pad, mode="edge")
    L = f.shape[axis]
    return sum(c[j] * np.take(fp, np.arange(n - 1 - j, n - 1 - j + L), axis=axis) for j in range(n))


def nlm_noise_sigma(img):
    # MAD estimate from the finest diagonal Haar sub-band
    return np.median(np.abs(pywt.dwt2(img, "haar")[1][2])) / 0.6745


def fcfds_kernel(img, alpha=NLM_ALPHA, n_taps=NLM_TAPS, phi_x=NLM_PHI_X, eta=None):
    c = fcfds_coeffs(alpha, n_taps)
    G = np.abs(frac_diff(img, c, axis=1)) + np.abs(frac_diff(img, c, axis=0))
    eta = G.mean() if eta is None else eta
    phi_y = 1.0 / (1.0 + G / eta)
    phi_pad = np.pad(phi_y, NLM_PATCH // 2, mode="edge")
    A = np.exp(-_OFFS[:, None, None] ** 2 / (2 * phi_pad[None]))
    return A, np.exp(-_OFFS ** 2 / (2 * phi_x))


def _nlm(v, A, gx, h, search=NLM_SEARCH):
    r, s = NLM_PATCH // 2, search // 2
    H, W = v.shape
    vp = np.pad(v, s + r, mode="edge").astype(np.float32)
    A, gx = A.astype(np.float32), gx.astype(np.float32)
    centre = vp[s:s + H + 2 * r, s:s + W + 2 * r]
    num = np.zeros((H, W), np.float32)
    den = np.zeros((H, W), np.float32)
    inv_h2 = np.float32(1.0 / h ** 2)
    for sy in range(-s, s + 1):
        for sx in range(-s, s + 1):
            cand = vp[s + sy:s + sy + H + 2 * r, s + sx:s + sx + W + 2 * r]
            D = (centre - cand) ** 2
            rows = np.zeros((H, W + 2 * r), np.float32)
            for i in range(NLM_PATCH):
                rows += (A[i] * D)[i:i + H]
            dist = np.zeros((H, W), np.float32)
            for i in range(NLM_PATCH):
                dist += gx[i] * rows[:, i:i + W]
            w = np.exp(-dist * inv_h2)
            num += w * cand[r:r + H, r:r + W]
            den += w
    return (num / den).astype(np.float64)


def nlm_denoise(img, h_factor=3.6, alpha=NLM_ALPHA):
    # h = h_f * sigma_MAD * sqrt(mean kernel mass), as in the NLM notebook
    img = np.asarray(img, np.float64)
    A, gx = fcfds_kernel(img, alpha=alpha)
    h = h_factor * nlm_noise_sigma(img) * np.sqrt(float(A.sum(axis=0).mean() * gx.sum()))
    return np.clip(_nlm(img, A, gx, h), 0, PEAK)


# ======================================================================================
# 3. Modified soft-threshold wavelet denoising, parameters chosen by PSO (Subhedar et al. 2023)
# ======================================================================================
def universal_threshold(sigma, M, K=1.0, d=1):
    return K * sigma * np.sqrt(2 * np.log(M)) / np.log(np.e + d - 1)


def modified_soft_threshold(y, y_th, alpha, beta):
    s = np.sign(y)
    x = y - s * y_th
    big = x + (2 / np.pi) * np.arctan(beta * x / y_th + s * alpha) * y_th
    small = (2 / np.pi) * np.arctan(alpha) * y
    return np.where(np.abs(y) >= y_th, big, small)


class WaveletDenoiser:
    # forward DWT computed once per image; only threshold + inverse DWT are re-run.
    # log_domain=True: the input is already log-compressed (the Duke .tifs are), so it is thresholded as stored.
    # log_domain=False: linear-intensity input, wrapped in log1p / expm1 as in the paper's Fig. 2.
    def __init__(self, img, wavelet="db2", level=1, log_domain=True):
        self.img, self.wavelet, self.level = np.asarray(img, np.float64), wavelet, level
        self.log_domain = log_domain
        self.coeffs = pywt.wavedec2(self.img if log_domain else np.log1p(self.img), wavelet, level=level)
        self.sigma = np.median(np.abs(self.coeffs[-1][2])) / 0.6745       # HH of level 1
        self.M = self.img.size

    def modified(self, alpha, beta, K):
        new = [self.coeffs[0]]
        for k, details in enumerate(self.coeffs[1:]):
            d = self.level - k
            th = universal_threshold(self.sigma, self.M, K, d)
            new.append(tuple(modified_soft_threshold(c, th, alpha, beta) for c in details))
        rec = pywt.waverec2(new, self.wavelet)[: self.img.shape[0], : self.img.shape[1]]
        return np.clip(rec if self.log_domain else np.expm1(rec), 0, PEAK)


def wavelet_mst(img, alpha, beta, K, level=1, wavelet="db2", log_domain=True):
    return WaveletDenoiser(img, wavelet, level, log_domain).modified(alpha, beta, K)


def pso_optimize(fitness, bounds, n_particles=100, n_iter=15, w=0.7298, c1=1.49618, c2=1.49618, seed=42):
    # Same PSO as the PSO notebook (Clerc-Kennedy constriction, velocity clipped to 20% of range)
    names = list(bounds)
    lo, hi = np.array([bounds[k] for k in names], dtype=float).T
    v_max = 0.2 * (hi - lo)
    rng = np.random.default_rng(seed)
    pos = rng.uniform(lo, hi, size=(n_particles, len(names)))
    vel = rng.uniform(-v_max, v_max, size=pos.shape)
    pbest_pos, pbest_fit = pos.copy(), np.full(n_particles, -np.inf)
    gbest_pos, gbest_fit = pos[0].copy(), -np.inf
    history = []
    for _ in range(n_iter):
        for i in range(n_particles):
            fit = fitness(**dict(zip(names, pos[i])))
            if fit > pbest_fit[i]:
                pbest_fit[i], pbest_pos[i] = fit, pos[i].copy()
            if fit > gbest_fit:
                gbest_fit, gbest_pos = fit, pos[i].copy()
            r1, r2 = rng.random(len(names)), rng.random(len(names))
            vel[i] = w * vel[i] + c1 * r1 * (pbest_pos[i] - pos[i]) + c2 * r2 * (gbest_pos - pos[i])
            vel[i] = np.clip(vel[i], -v_max, v_max)
            pos[i] = np.clip(pos[i] + vel[i], lo, hi)
        history.append(gbest_fit)
    return dict(zip(names, gbest_pos)), gbest_fit, history


def pso_tune_wavelet(level, images, refs, bounds, n_particles=30, n_iter=15, seed=42, log_domain=True):
    """PSO over (alpha, beta, K) at one decomposition level; fitness = mean PSNR against the references."""
    dens = [WaveletDenoiser(z, "db2", level, log_domain) for z in images]

    def fitness(**p):
        return float(np.mean([psnr(r, d.modified(**p)) for d, r in zip(dens, refs)]))

    params, best, history = pso_optimize(fitness, bounds, n_particles, n_iter, seed=seed)
    return {"level": level, **params, "fitness_PSNR": best, "history": history}


# ======================================================================================
# 4. Sparse representation with a K-SVD global dictionary + noise estimation (Zhang et al. 2022)
# ======================================================================================
KSVD_PATCH, KSVD_MAX_ATOMS, KSVD_LAMBDA_SCALE = 8, 12, 30.0
_N_KERNEL = np.array([[1., -2., 1.], [-2., 4., -2.], [1., -2., 1.]])       # Eq. (5), Immerkaer


def estimate_noise_sigma(block):
    # Eq. (7): sigma = sqrt(pi/2) / (6 (W-2)(H-2)) * sum |I * N|
    r = convolve2d(np.asarray(block, np.float64), _N_KERNEL, mode="valid")
    h, w = r.shape
    return float(np.sqrt(np.pi / 2) * np.abs(r).sum() / (6.0 * (w - 2) * (h - 2)))


def glcm_entropy(block, levels=32):
    q = np.clip((np.asarray(block, np.float64) / PEAK * levels).astype(np.int32), 0, levels - 1)
    g = graycomatrix(q, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                     levels=levels, symmetric=True, normed=True).astype(np.float64)
    p = g + 1e-12
    return float(-(p * np.log2(p)).sum() / g.shape[3])


def weak_texture_sigma(img, block=64, min_std=2.0):
    # Eq. (6)+(7): noise estimated on the minimum-GLCM-entropy tile
    best = None
    H, W = img.shape
    for y in range(0, H - block + 1, block):
        for x in range(0, W - block + 1, block):
            tile = np.asarray(img[y:y + block, x:x + block], np.float64)
            if tile.std() < min_std:
                continue
            e = glcm_entropy(tile)
            if best is None or e < best[0]:
                best = (e, tile)
    return estimate_noise_sigma(best[1])


def dct_dictionary(patch_size=8, n_atoms=256):
    n = patch_size * patch_size
    n_freq = int(np.ceil(np.sqrt(n_atoms)))
    p = np.arange(patch_size)
    B = np.cos(np.pi * np.outer(np.arange(n_freq) / 2.0, 2 * p + 1) / (2.0 * patch_size))
    B /= np.linalg.norm(B, axis=1, keepdims=True)
    D = np.zeros((n, n_atoms))
    k = 0
    for i in range(n_freq):
        for j in range(n_freq):
            if k >= n_atoms:
                break
            atom = np.outer(B[i], B[j]).ravel()
            D[:, k] = atom / np.linalg.norm(atom)
            k += 1
    return D


def extract_patches(img, patch_size=8):
    win = sliding_window_view(np.asarray(img, np.float64), (patch_size, patch_size))
    h, w = win.shape[:2]
    return win.reshape(h * w, patch_size * patch_size), (h, w)


def patches_to_image(P, grid, patch_size=8):
    h, w = grid
    P = np.asarray(P).reshape(h, w, patch_size, patch_size)
    acc = np.zeros((h + patch_size - 1, w + patch_size - 1))
    cnt = np.zeros_like(acc)
    for dy in range(patch_size):
        for dx in range(patch_size):
            acc[dy:dy + h, dx:dx + w] += P[:, :, dy, dx]
            cnt[dy:dy + h, dx:dx + w] += 1.0
    return acc, cnt


def omp(D, Y, n_nonzero=12, tol=None, batch=8192):
    # Batched OMP of the K-SVD notebook (Eq. 2): stops at n_nonzero atoms or ||r||^2 <= tol
    n, K = D.shape
    N = Y.shape[1]
    G = D.T @ D
    vals = np.zeros((N, n_nonzero))
    sup = np.full((N, n_nonzero), -1, np.int32)
    for start in range(0, N, batch):
        stop = min(start + batch, N)
        Yb = Y[:, start:stop]
        B = stop - start
        S = np.full((B, n_nonzero), -1, np.int32)
        A = np.zeros((B, n_nonzero))
        r = Yb.copy()
        active = np.ones(B, bool)
        for t in range(n_nonzero):
            ia = np.flatnonzero(active)
            if ia.size == 0:
                break
            corr = np.abs(r[:, ia].T @ D)
            for j in range(t):
                col = S[ia, j]
                good = col >= 0
                corr[np.flatnonzero(good), col[good]] = -1.0
            S[ia, t] = corr.argmax(1)
            idx = S[ia, :t + 1]
            Gs = G[idx[:, :, None], idx[:, None, :]] + 1e-9 * np.eye(t + 1)
            Dsel = D.T[idx]
            b = np.einsum("bmn,bn->bm", Dsel, Yb[:, ia].T)
            a = np.linalg.solve(Gs, b[..., None])[..., 0]
            A[ia, :t + 1] = a
            r[:, ia] = Yb[:, ia] - np.einsum("bm,bmn->bn", a, Dsel).T
            if tol is not None:
                active[ia] = np.einsum("ij,ij->j", r[:, ia], r[:, ia]) > tol
        vals[start:stop] = A
        sup[start:stop] = S
    return vals, sup


def load_global_dictionary(path=KSVD_CHECKPOINT):
    # 64x256 dictionary learned by K-SVD (20 sweeps) on averaged images of subjects 1-12
    return np.load(path)["D"]


def ksvd_denoise(img, D, omp_C=1.15, lambda_scale=KSVD_LAMBDA_SCALE, sigma=None):
    img = np.asarray(img, np.float64)
    sigma = weak_texture_sigma(img) if sigma is None else sigma
    pad = KSVD_PATCH // 2
    padded = np.pad(img, pad, mode="reflect")
    P, grid = extract_patches(padded, KSVD_PATCH)
    tol = (omp_C * sigma * np.sqrt(KSVD_PATCH ** 2)) ** 2
    vals, sup = omp(D, P.T, n_nonzero=KSVD_MAX_ATOMS, tol=tol)
    est = np.einsum("bm,bmn->bn", vals, D.T[np.clip(sup, 0, D.shape[1] - 1)])
    acc, cnt = patches_to_image(est, grid, KSVD_PATCH)
    lam = lambda_scale / max(sigma, 1e-8)
    X = ((lam * padded + acc) / (lam + cnt))[pad:-pad, pad:-pad]          # Eq. (3)
    return np.clip(X, 0, PEAK)


# ======================================================================================
# Dispatch — a top-level function so the notebook can fan runs out over a process pool
# ======================================================================================
_DICT = None


def denoise(method, img, params):
    """Run one method with one parameter set; returns (denoised image, seconds)."""
    global _DICT
    t0 = time.perf_counter()
    if method == "TGV":
        out = tgv_denoise(img, **params)
    elif method == "NLM":
        out = nlm_denoise(img, **params)
    elif method == "Wavelet":
        out = wavelet_mst(img, **params)
    elif method == "KSVD":
        if _DICT is None:
            _DICT = load_global_dictionary()
        out = ksvd_denoise(img, _DICT, **params)
    else:
        raise ValueError(method)
    return out, time.perf_counter() - t0


def score(method, img, ref, params):
    """Denoise and score against the reference; returns (metrics dict, seconds)."""
    out, dt = denoise(method, img, params)
    return all_metrics(ref, out), dt
