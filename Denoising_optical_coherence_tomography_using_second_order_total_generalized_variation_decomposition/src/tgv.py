"""
Second-order total-generalised-variation (TGV^2) decomposition denoising,
after Duan, Lu, Tench, Gottlob, Proudlock, Samani & Bai, "Denoising optical coherence
tomography using second order total generalized variation decomposition",
Biomed. Signal Process. Control 24 (2016) 120-127.

The model solved is Eq. (2.11) of the paper,

    min_{u,g,p} 1/2||f - u - div g||_2^2 + beta*||g||_1
                + alpha1*||grad u - p||_1 + alpha0*||eps(p)||_1 ,

via the split-Bregman iteration of Eq. (3.1) / Algorithm 1.  Every closed-form update
below is *derived from (3.1)* (see report/paper_summary.md, section 1.3) and is checked
against a direct sparse solve in ``tests/test_operators.py``.

Sign conventions (see operators.py): grad = forward differences, div = backward
differences = -grad^T, periodic boundaries, h = 1.

Author: reproduction study.  Licensed under MIT.
"""

from __future__ import annotations

import time

import numpy as np

from .operators import (
    div, div_sym, fft_symbols, grad, isotropic_shrink, modulus, sym_grad,
)

__all__ = ["tgv_decompose", "vo_decompose", "energy_terms"]


def _f(x):
    return np.fft.fft2(x)


def _if(x):
    return np.real(np.fft.ifft2(x))


# --------------------------------------------------------------------------------------
# the three FFT closed-form updates, factored out so that they can be unit-tested
# against a direct sparse solve (see src/verify.py)
# --------------------------------------------------------------------------------------
def update_u(f, g, w, p, b2, lam2, sym):
    """Eq. (3.2): (1 - lam2*Delta) u = f - div g - lam2 div(w + p - b2)."""
    rhs = f - div(g)
    return _if(_f(rhs - lam2 * div((w[0] + p[0] - b2[0], w[1] + p[1] - b2[1])))
               / sym["den_u"])


def update_g(f, u, m, b1, lam1, sym):
    """Eqs. (3.3)-(3.4): (lam1*I - grad div) g = lam1 (m - b1) + grad(u - f)."""
    ex, ey = grad(u - f)
    H1 = _f(ex + lam1 * (m[0] - b1[0]))
    H2 = _f(ey + lam1 * (m[1] - b1[1]))
    a11, a12, a21, a22, D = (sym[k] for k in ("a11", "a12", "a21", "a22", "D"))
    return (_if((a22 * H1 - a12 * H2) / D),
            _if((-a21 * H1 + a11 * H2) / D))


def update_p(u, w, b2, v, b3, lam2, lam3, sym):
    """Eqs. (3.5)-(3.6): (lam2*I + lam3 eps^T eps) p = lam2(grad u + b2 - w)
    - lam3 div_sym(v - b3)."""
    ux, uy = grad(u)
    d1, d2 = div_sym((v[0] - b3[0], v[1] - b3[1], v[2] - b3[2]))
    R1 = _f(lam2 * (ux + b2[0] - w[0]) - lam3 * d1)
    R2 = _f(lam2 * (uy + b2[1] - w[1]) - lam3 * d2)
    b11, b12, b21, b22, B = (sym[k] for k in ("b11", "b12", "b21", "b22", "B"))
    return (_if((b22 * R1 - b12 * R2) / B),
            _if((-b21 * R1 + b11 * R2) / B))


# --------------------------------------------------------------------------------------
# TGV^2 split Bregman
# --------------------------------------------------------------------------------------
def tgv_decompose(f, alpha0, alpha1, beta, lam1, lam2, lam3, max_iter=300,
                  tol=1e-3, return_info=False):
    """Split-Bregman TGV^2 decomposition of ``f`` into structure ``u`` and texture.

    Parameters
    ----------
    f : (M, N) float array
        Input image in the *log* domain (additive-noise assumption), nominally in [0, 1].
    alpha0, alpha1 : float
        TGV weights on ||eps(p)||_1 and ||grad u - p||_1 respectively.
    beta : float
        Weight on the texture L1 norm ||g||_1  (paper's beta, Eq. 2.6).
    lam1, lam2, lam3 : float
        Split-Bregman penalty parameters for the constraints m=g, w=grad u-p, v=eps(p).
    max_iter : int
        Hard iteration cap.
    tol : float
        Stop when ||u_k - u_{k-1}||_2 / ||u_k||_2 < tol (the paper uses 1e-3).
    return_info : bool
        If True also return a dict with per-iteration diagnostics, iteration count and
        wall-clock time.

    Returns
    -------
    u : structure (denoised) image
    texture : f - u, the estimated speckle/noise field  (== div(g) at convergence)
    info : dict, only when ``return_info`` is True
    """
    f = np.asarray(f, dtype=np.float64)
    if f.ndim != 2:
        raise ValueError("tgv_decompose expects a 2-D array")
    for name, val in (("alpha0", alpha0), ("alpha1", alpha1), ("beta", beta),
                      ("lam1", lam1), ("lam2", lam2), ("lam3", lam3)):
        if not np.isfinite(val) or val < 0:
            raise ValueError(f"{name} must be finite and >= 0, got {val}")

    sym = fft_symbols(f.shape, lam1, lam2, lam3)

    # ---- initialisation: every primal/dual/split/Bregman variable = 0 (Algorithm 1) ----
    u = np.zeros_like(f)
    g1 = np.zeros_like(f); g2 = np.zeros_like(f)
    p1 = np.zeros_like(f); p2 = np.zeros_like(f)
    v11 = np.zeros_like(f); v22 = np.zeros_like(f); v12 = np.zeros_like(f)
    m1 = np.zeros_like(f); m2 = np.zeros_like(f)
    w1 = np.zeros_like(f); w2 = np.zeros_like(f)
    b1_1 = np.zeros_like(f); b1_2 = np.zeros_like(f)
    b2_1 = np.zeros_like(f); b2_2 = np.zeros_like(f)
    b3_11 = np.zeros_like(f); b3_22 = np.zeros_like(f); b3_12 = np.zeros_like(f)

    hist = {"rel_change": [], "energy_aug": [], "energy_model": [],
            "data_term": [], "tv_g": [], "tv_w": [], "tv_v": []}

    t0 = time.perf_counter()
    n_iter = 0
    for it in range(1, max_iter + 1):
        u_prev = u
        g = (g1, g2)

        # ---- (3.2) u : (1 - lam2*Delta) u = f - div g - lam2 div(w + p - b2) ----------
        u = update_u(f, g, (w1, w2), (p1, p2), (b2_1, b2_2), lam2, sym)

        # ---- (3.3)-(3.4) g : A g = lam1 (m - b1) + grad(u - f) -----------------------
        g1, g2 = update_g(f, u, (m1, m2), (b1_1, b1_2), lam1, sym)

        # ---- (3.5)-(3.6) p : (lam2 I + lam3 eps^T eps) p = lam2(grad u + b2 - w)
        #                     - lam3 div_sym(v - b3) ------------------------------------
        p1, p2 = update_p(u, (w1, w2), (b2_1, b2_2),
                          (v11, v22, v12), (b3_11, b3_22, b3_12), lam2, lam3, sym)

        # ---- (3.7) v : isotropic shrinkage of eps(p) + b3 ----------------------------
        e11, e22, e12 = sym_grad((p1, p2))
        v11, v22, v12 = isotropic_shrink(
            (e11 + b3_11, e22 + b3_22, e12 + b3_12), alpha0 / lam3, weights=(1, 1, 2))

        # ---- (3.8) m : isotropic shrinkage of g + b1 --------------------------------
        m1, m2 = isotropic_shrink((g1 + b1_1, g2 + b1_2), beta / lam1)

        # ---- (3.9) w : isotropic shrinkage of grad u - p + b2 -----------------------
        ux, uy = grad(u)
        w1, w2 = isotropic_shrink((ux - p1 + b2_1, uy - p2 + b2_2), alpha1 / lam2)

        # ---- Bregman updates (Algorithm 1, steps 10-12) -----------------------------
        b1_1 = b1_1 + (g1 - m1)
        b1_2 = b1_2 + (g2 - m2)
        b2_1 = b2_1 + (ux - p1 - w1)
        b2_2 = b2_2 + (uy - p2 - w2)
        b3_11 = b3_11 + (e11 - v11)
        b3_22 = b3_22 + (e22 - v22)
        b3_12 = b3_12 + (e12 - v12)

        # ---- diagnostics ------------------------------------------------------------
        nu = np.linalg.norm(u)
        rel = np.linalg.norm(u - u_prev) / (nu if nu > 0 else 1.0)
        et = energy_terms(f, u, (g1, g2), (p1, p2), (m1, m2), (w1, w2),
                          (v11, v22, v12), (b1_1, b1_2), (b2_1, b2_2),
                          (b3_11, b3_22, b3_12), alpha0, alpha1, beta, lam1, lam2, lam3)
        hist["rel_change"].append(rel)
        for k, val in et.items():
            hist[k].append(val)
        n_iter = it
        if rel < tol:
            break

    elapsed = time.perf_counter() - t0
    texture = f - u
    if not return_info:
        return u, texture
    hist["n_iter"] = n_iter
    hist["time_s"] = elapsed
    hist["rel_change"] = np.asarray(hist["rel_change"])
    for k in ("energy_aug", "energy_model", "data_term", "tv_g", "tv_w", "tv_v"):
        hist[k] = np.asarray(hist[k])
    return u, texture, hist


# --------------------------------------------------------------------------------------
# Vese-Osher (TV) decomposition : the same solver with TGV replaced by TV
# --------------------------------------------------------------------------------------
def vo_decompose(f, alpha1, beta, lam1, lam2, max_iter=300, tol=1e-3, return_info=False):
    """Vese-Osher decomposition, Eq. (2.5):

        min_{u,g} 1/2||f - u - div g||^2 + beta||g||_1 + alpha1||grad u||_1

    This is exactly ``tgv_decompose`` with ``p == 0`` fixed (no deformation field, no
    ``v``/``b3`` variable), i.e. the TGV term replaced by TV.  ``lam3``/``alpha0`` are
    irrelevant and therefore absent from the signature.
    """
    f = np.asarray(f, dtype=np.float64)
    sym = fft_symbols(f.shape, lam1, lam2, 1.0)
    den_u, a11, a12, a21, a22, D = (sym[k] for k in
                                    ("den_u", "a11", "a12", "a21", "a22", "D"))
    F = _f

    u = np.zeros_like(f)
    g1 = np.zeros_like(f); g2 = np.zeros_like(f)
    m1 = np.zeros_like(f); m2 = np.zeros_like(f)
    w1 = np.zeros_like(f); w2 = np.zeros_like(f)
    b1_1 = np.zeros_like(f); b1_2 = np.zeros_like(f)
    b2_1 = np.zeros_like(f); b2_2 = np.zeros_like(f)

    hist = {"rel_change": [], "energy_model": [], "data_term": [],
            "tv_g": [], "tv_w": []}
    t0 = time.perf_counter()
    n_iter = 0
    for it in range(1, max_iter + 1):
        u_prev = u
        u = _if(F(f - div((g1, g2)) - lam2 * div((w1 - b2_1, w2 - b2_2))) / den_u)

        ex, ey = grad(u - f)
        h1, h2 = ex + lam1 * (m1 - b1_1), ey + lam1 * (m2 - b1_2)
        H1, H2 = F(h1), F(h2)
        g1 = _if((a22 * H1 - a12 * H2) / D)
        g2 = _if((-a21 * H1 + a11 * H2) / D)

        ux, uy = grad(u)
        m1, m2 = isotropic_shrink((g1 + b1_1, g2 + b1_2), beta / lam1)
        w1, w2 = isotropic_shrink((ux + b2_1, uy + b2_2), alpha1 / lam2)

        b1_1, b1_2 = b1_1 + (g1 - m1), b1_2 + (g2 - m2)
        b2_1, b2_2 = b2_1 + (ux - w1), b2_2 + (uy - w2)

        nu = np.linalg.norm(u)
        rel = np.linalg.norm(u - u_prev) / (nu if nu > 0 else 1.0)
        hist["rel_change"].append(rel)
        hist["energy_model"].append(
            0.5 * np.sum((f - u - div((g1, g2))) ** 2)
            + beta * np.sum(modulus((g1, g2), )) + alpha1 * np.sum(modulus((ux, uy))))
        hist["data_term"].append(0.5 * np.sum((f - u - div((g1, g2))) ** 2))
        hist["tv_g"].append(beta * np.sum(modulus((g1, g2))))
        hist["tv_w"].append(alpha1 * np.sum(modulus((ux, uy))))
        n_iter = it
        if rel < tol:
            break

    elapsed = time.perf_counter() - t0
    if not return_info:
        return u, f - u
    hist["n_iter"] = n_iter
    hist["time_s"] = elapsed
    for k in ("rel_change", "energy_model", "data_term", "tv_g", "tv_w"):
        hist[k] = np.asarray(hist[k])
    return u, f - u, hist


# --------------------------------------------------------------------------------------
# energy of the augmented functional (3.1) and of the original model (2.11)
# --------------------------------------------------------------------------------------
def energy_terms(f, u, g, p, m, w, v, b1, b2, b3, alpha0, alpha1, beta,
                 lam1, lam2, lam3):
    """Evaluate every term of (3.1) plus the data/TV terms of (2.11) for diagnostics."""
    g1, g2 = g
    p1, p2 = p
    m1, m2 = m
    w1, w2 = w
    v11, v22, v12 = v
    b1_1, b1_2 = b1
    b2_1, b2_2 = b2
    b3_11, b3_22, b3_12 = b3

    r = f - u - div((g1, g2))
    data = 0.5 * float(np.sum(r ** 2))
    tv_g = beta * float(np.sum(modulus((m1, m2))))
    tv_w = alpha1 * float(np.sum(modulus((w1, w2))))
    tv_v = alpha0 * float(np.sum(modulus((v11, v22, v12), (1, 1, 2))))

    pen = 0.5 * lam1 * float(np.sum((m1 - g1 - b1_1) ** 2 + (m2 - g2 - b1_2) ** 2))
    ux, uy = grad(u)
    pen += 0.5 * lam2 * float(np.sum((w1 - ux + p1 - b2_1) ** 2
                                     + (w2 - uy + p2 - b2_2) ** 2))
    e11, e22, e12 = sym_grad((p1, p2))
    pen += 0.5 * lam3 * float(np.sum((v11 - e11 - b3_11) ** 2
                                     + (v22 - e22 - b3_22) ** 2
                                     + 2.0 * (v12 - e12 - b3_12) ** 2))

    # original (2.11) objective evaluated at the *recovered* variables
    model = (0.5 * float(np.sum(r ** 2))
             + beta * float(np.sum(modulus((g1, g2))))
             + alpha1 * float(np.sum(modulus((ux - p1, uy - p2))))
             + alpha0 * float(np.sum(modulus(
                 (e11, e22, e12), (1, 1, 2)))))

    return {"energy_aug": data + tv_g + tv_w + tv_v + pen,
            "energy_model": model,
            "data_term": data, "tv_g": tv_g, "tv_w": tv_w, "tv_v": tv_v}
