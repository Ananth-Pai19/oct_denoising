"""
Discrete differential operators and their exact FFT symbols.

Conventions (identical to the paper, Sec. 2.3, and to the numerical implementation):

    * a 2-D image is an array of shape (M, N) with axis 0 = y (rows) and axis 1 = x (columns)
    * ``grad``   uses *forward* differences
    * ``div``    uses *backward* differences, so that ``div = -grad^T`` exactly
                 (periodic boundary conditions, h = 1)
    * ``sym_grad`` is the symmetrised gradient eps(p) = (grad p + grad p^T)/2, stored as the
      three independent components (eps_11, eps_22, eps_12) of a symmetric 2x2 tensor;
      the pointwise Frobenius modulus of the full 2x2 tensor is
      sqrt(eps_11^2 + eps_22^2 + 2*eps_12^2)
    * ``div_sym`` is the (positive) divergence of a symmetric tensor field,
      div_sym(s) = (d_x s_11 + d_y s_12, d_x s_12 + d_y s_22) with backward differences.
      It satisfies  eps^T = -div_sym  (see ``sym_grad_adjoint``).

All pointwise vector quantities are represented as tuples of arrays that can be zipped
together; helper ``modulus`` returns the isotropic pointwise 2-norm.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "grad", "div", "sym_grad", "sym_grad_adjoint", "div_sym",
    "modulus", "isotropic_shrink", "laplacian", "fft_symbols",
]


# --------------------------------------------------------------------------------------
# spatial-domain operators (periodic, h = 1)
# --------------------------------------------------------------------------------------
def grad(u: np.ndarray):
    """Forward-difference gradient: returns (u[i,j+1]-u[i,j], u[i+1,j]-u[i,j])."""
    return (np.roll(u, -1, axis=1) - u, np.roll(u, -1, axis=0) - u)


def _grad_T(g):
    """Adjoint of ``grad`` (= -div)."""
    g1, g2 = g
    return -(g1 - np.roll(g1, 1, axis=1)) - (g2 - np.roll(g2, 1, axis=0))


def div(g: tuple) -> np.ndarray:
    """Backward-difference divergence ``div = -grad^T`` (exact negative adjoint of grad)."""
    g1, g2 = g
    return (g1 - np.roll(g1, 1, axis=1)) + (g2 - np.roll(g2, 1, axis=0))


def sym_grad(p: tuple):
    """Symmetric gradient eps(p), returned as (e11, e22, e12); e21 == e12."""
    p1, p2 = p
    dx_p1 = np.roll(p1, -1, axis=1) - p1
    dy_p1 = np.roll(p1, -1, axis=0) - p1
    dx_p2 = np.roll(p2, -1, axis=1) - p2
    dy_p2 = np.roll(p2, -1, axis=0) - p2
    return (dx_p1, dy_p2, 0.5 * (dy_p1 + dx_p2))


def div_sym(s: tuple):
    """Divergence of a symmetric tensor field, backward differences, positive sign."""
    s11, s22, s12 = s
    dx_s11 = s11 - np.roll(s11, 1, axis=1)
    dy_s12 = s12 - np.roll(s12, 1, axis=0)
    dx_s12 = s12 - np.roll(s12, 1, axis=1)
    dy_s22 = s22 - np.roll(s22, 1, axis=0)
    return (dx_s11 + dy_s12, dx_s12 + dy_s22)


def sym_grad_adjoint(s: tuple):
    """eps^T(s) = -div_sym(s)."""
    a, b = div_sym(s)
    return (-a, -b)


def laplacian(u: np.ndarray) -> np.ndarray:
    """Standard 5-point Laplacian = div(grad(u)) with the operators above."""
    return div(grad(u))


# --------------------------------------------------------------------------------------
# pointwise (isotropic) geometry
# --------------------------------------------------------------------------------------
def modulus(parts: tuple, weights: tuple | None = None) -> np.ndarray:
    """Isotropic pointwise L2 modulus of a vector/tensor field.

    ``weights`` defaults to (1,1,...,1).  For a symmetric tensor stored as
    (s11, s22, s12) pass weights=(1,1,2) so that the result equals the Frobenius norm of
    the full 2x2 matrix (the off-diagonal appears twice).
    """
    if weights is None:
        weights = (1.0,) * len(parts)
    acc = np.zeros_like(parts[0], dtype=np.float64)
    for p, w in zip(parts, weights):
        acc += w * p * p
    return np.sqrt(acc)


def isotropic_shrink(z: tuple, thresh: float, weights: tuple | None = None):
    """Pointwise isotropic soft-threshold: z/|z| * max(|z|-thresh, 0), 0*(0/0)=0."""
    mag = modulus(z, weights)
    scale = np.zeros_like(mag)
    nz = mag > 0
    scale[nz] = np.maximum(mag[nz] - thresh, 0.0) / mag[nz]
    return tuple(scale * zc for zc in z)


# --------------------------------------------------------------------------------------
# FFT symbols
# --------------------------------------------------------------------------------------
def fft_symbols(shape, lam1: float, lam2: float, lam3: float) -> dict:
    """Fourier symbols of every operator/matrix needed by the split-Bregman sweeps.

    For the forward difference along x the symbol is ``dx = exp(i kx) - 1``; the backward
    difference used by ``div`` has symbol ``conj(dx) = -dx_forward``... more precisely
    ``1 - exp(-i kx) = -conj(dx)``.  The Laplacian symbol is therefore
    ``F(Delta) = -(|dx|^2 + |dy|^2)``.

    Returns a dict with the pointwise (M,N) arrays used by :mod:`tgv`.
    """
    m, n = shape
    kx = 2.0 * np.pi * np.fft.fftfreq(n)      # columns (axis 1)
    ky = 2.0 * np.pi * np.fft.fftfreq(m)      # rows    (axis 0)
    KX, KY = np.meshgrid(kx, ky)

    dx = np.exp(1j * KX) - 1.0                # forward difference along x
    dy = np.exp(1j * KY) - 1.0                # forward difference along y
    cbx = np.abs(dx) ** 2                     # |symbol(backward_x)|^2
    cby = np.abs(dy) ** 2

    sym = {"dx": dx, "dy": dy, "abs_dx2": cbx, "abs_dy2": cby}

    # --- u-subproblem: (1 - lam2 * F(Delta)) -------------------------------------------
    sym["den_u"] = 1.0 + lam2 * (cbx + cby)

    # --- g-subproblem: A = lam1*I - grad(div), Hermitian 2x2 ---------------------------
    sym["a11"] = lam1 + cbx
    sym["a22"] = lam1 + cby
    sym["a12"] = dx * np.conj(dy)             # F(-d_x^B d_y^F) with B backward, F forward
    sym["a21"] = np.conj(dx) * dy             # = conj(a12)
    sym["D"] = sym["a11"] * sym["a22"] - sym["a12"] * sym["a21"]

    # --- p-subproblem: Bmat = lam2*I - lam3*div_sym(eps), Hermitian 2x2 ----------------
    #     symbol of (-div_sym o eps)_11 = |dx|^2 + |dy|^2/2, _22 = |dx|^2/2 + |dy|^2,
    #     off-diagonal = -(1/2) d_y^B d_x^F = +(1/2) dx conj(dy).  This reproduces the
    #     paper's printed F(lam2 - lam3 d_x d_x - (lam3/2) d_y d_y) and F(-(lam3/2) d_y d_x)
    #     with d_x,d_y the *backward* differences of eq. (3.5)-(3.6).
    sym["b11"] = lam2 + lam3 * cbx + 0.5 * lam3 * cby
    sym["b22"] = lam2 + 0.5 * lam3 * cbx + lam3 * cby
    sym["b12"] = 0.5 * lam3 * dx * np.conj(dy)
    sym["b21"] = np.conj(sym["b12"])
    sym["B"] = sym["b11"] * sym["b22"] - sym["b12"] * sym["b21"]

    return sym
