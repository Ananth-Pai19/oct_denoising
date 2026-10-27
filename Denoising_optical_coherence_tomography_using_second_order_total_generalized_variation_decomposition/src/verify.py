"""
Numerical verification of the discrete operators and of the three FFT closed-form
updates used by the split-Bregman TGV solver.

Two levels of checking:

1. *Adjoint consistency* - the discrete operators must satisfy
   ``div == -grad^T`` and ``sym_grad_adjoint == -div_sym^T`` exactly (periodic BC, h=1).

2. *Closed-form vs. direct solve* - each of the three quadratic sub-problems of the
   split-Bregman sweep is assembled as an explicit sparse matrix and solved with
   ``scipy.sparse.linalg.spsolve``.  The FFT closed form must reproduce that solution to
   machine precision.  This is what validates the derived equations (3.2)-(3.6)
   independently of how the garbled PDF equations were typeset.

Run ``python -m src.verify`` for a human-readable report.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from .operators import div, div_sym, fft_symbols, grad, sym_grad, sym_grad_adjoint
from .tgv import update_g, update_p, update_u

__all__ = ["check_adjoints", "check_closed_forms", "report"]


# --------------------------------------------------------------------------------------
# explicit operator matrices built by applying the operators to unit vectors
# --------------------------------------------------------------------------------------
def _mat(op, shape, n_out, n_in=1):
    """Matrix of a linear operator that maps n_in component-images -> n_out images."""
    m, n = shape
    mn = m * n
    A = np.zeros((n_out * mn, n_in * mn))
    for j in range(n_in * mn):
        e = np.zeros(mn)
        e[j % mn] = 1.0
        inp = tuple(e.reshape(shape) if k == j // mn else np.zeros(shape)
                    for k in range(n_in))
        cols = op(inp[0]) if n_in == 1 else op(inp)
        if not isinstance(cols, tuple):
            cols = (cols,)
        for k in range(n_out):
            A[k * mn:(k + 1) * mn, j] = cols[k].ravel()
    return A


def operators(shape):
    """Return dense matrices (G, Dv, E, DS) for ``grad, div, sym_grad, div_sym``."""
    m, n = shape
    mn = m * n
    G = _mat(grad, shape, 2)
    Dv = _mat(div, shape, 1, n_in=2)
    E = _mat(sym_grad, shape, 3, n_in=2)     # (3mn, mn)  rows: e11, e22, e12
    DS = _mat(div_sym, shape, 2, n_in=3)     # (2mn, 3mn)
    return G, Dv, E, DS


def check_adjoints(shape=(12, 16), seed=0):
    rng = np.random.default_rng(seed)
    G, Dv, E, DS = operators(shape)
    mn = shape[0] * shape[1]

    u = rng.standard_normal(shape)
    g = (rng.standard_normal(shape), rng.standard_normal(shape))
    lhs = float(np.sum(div(g) * u))
    rhs = -float(np.sum(np.concatenate([a.ravel() for a in g]) * (G @ u.ravel())))
    err_div = abs(lhs - rhs) / max(abs(lhs), 1e-30)

    p = (rng.standard_normal(shape), rng.standard_normal(shape))
    s = (rng.standard_normal(shape), rng.standard_normal(shape),
         rng.standard_normal(shape))
    # <eps p, s>_F with the off-diagonal of the symmetric tensor counted twice
    Ep = E @ np.concatenate([a.ravel() for a in p])
    sv = np.concatenate([a.ravel() for a in s])
    w3 = np.concatenate([np.ones(shape[0] * shape[1]),
                         np.ones(shape[0] * shape[1]),
                         2.0 * np.ones(shape[0] * shape[1])])
    lhs2 = float(np.sum(w3 * sv * Ep))
    # <p, eps^T s> with eps^T s = -div_sym(s)
    adj_s = sym_grad_adjoint(s)
    rhs2 = float(np.sum(np.concatenate([a.ravel() for a in p])
                        * np.concatenate([a.ravel() for a in adj_s])))
    err_sym = abs(lhs2 - rhs2) / max(abs(lhs2), 1e-30)

    return {"div_adjoint_rel_err": err_div,
            "sym_grad_adjoint_rel_err": err_sym,
            "max_abs_div_minus_negGt": float(np.abs(Dv + G.T).max())}


# --------------------------------------------------------------------------------------
# closed form vs. direct sparse solve
# --------------------------------------------------------------------------------------
def check_closed_forms(shape=(10, 14), alpha0=0.05, alpha1=0.1, beta=0.02,
                       lam1=5.0, lam2=5.0, lam3=5.0, seed=1):
    rng = np.random.default_rng(seed)
    m, n = shape
    mn = m * n
    rd = lambda: rng.standard_normal(shape)     # noqa: E731

    f = rd()
    g = (rd(), rd())
    p = (rd(), rd())
    w = (rd(), rd())
    b2 = (rd(), rd())
    v = (rd(), rd(), rd())
    b3 = (rd(), rd(), rd())
    m_ = (rd(), rd())
    b1 = (rd(), rd())

    sym = fft_symbols(shape, lam1, lam2, lam3)
    G, Dv, E, DS = operators(shape)
    I = np.eye(mn)
    w3 = np.concatenate([np.ones(mn), np.ones(mn), 2.0 * np.ones(mn)])

    # ---- u-subproblem:  (I + lam2 G^T G) u = f - Dv g + lam2 G^T (w + p - b2) ---------
    GtG = G.T @ G
    rhs_u = (f.ravel() - Dv @ np.concatenate([a.ravel() for a in g])
             + lam2 * G.T @ np.concatenate([(w[0] + p[0] - b2[0]).ravel(),
                                            (w[1] + p[1] - b2[1]).ravel()]))
    u_direct = spsolve(sp.csc_matrix(I + lam2 * GtG), rhs_u).reshape(shape)
    u_fft = update_u(f, g, w, p, b2, lam2, sym)

    # ---- g-subproblem: (lam1 I + Dv^T Dv) g = lam1 (m - b1) + G (u - f) --------------
    DtD = Dv.T @ Dv
    rhs_g = (lam1 * np.concatenate([(m_[0] - b1[0]).ravel(), (m_[1] - b1[1]).ravel()])
             + G @ (u_direct - f).ravel())
    g_direct = spsolve(sp.csc_matrix(lam1 * np.eye(2 * mn) + DtD), rhs_g)
    g_fft = update_g(f, u_direct, m_, b1, lam1, sym)
    g_fft_v = np.concatenate([g_fft[0].ravel(), g_fft[1].ravel()])

    # ---- p-subproblem: (lam2 I - lam3 div_sym eps) p
    #        = lam2 (G u + b2 - w) - lam3 div_sym(v - b3)   [weighted Frobenius norm] ---
    EWE = E.T @ (w3[:, None] * E)        # eps^T W eps
    t = (v[0] - b3[0], v[1] - b3[1], v[2] - b3[2])
    rhs_p = (lam2 * np.concatenate([(np.asarray(grad(u_direct)[0]) + b2[0] - w[0]).ravel(),
                                    (np.asarray(grad(u_direct)[1]) + b2[1] - w[1]).ravel()])
             - lam3 * DS @ np.concatenate([a.ravel() for a in t]))
    p_direct = spsolve(sp.csc_matrix(lam2 * np.eye(2 * mn) + lam3 * EWE), rhs_p)
    p_fft = update_p(u_direct, w, b2, v, b3, lam2, lam3, sym)
    p_fft_v = np.concatenate([p_fft[0].ravel(), p_fft[1].ravel()])

    rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))  # noqa: E731
    return {"u_rel_err": rel(u_fft.ravel(), u_direct.ravel()),
            "g_rel_err": rel(g_fft_v, g_direct),
            "p_rel_err": rel(p_fft_v, p_direct)}


def report():
    adj = check_adjoints()
    cf = check_closed_forms()
    print("== adjoint consistency (periodic BC, h=1) ==")
    for k, v in adj.items():
        print(f"  {k:28s} {v:.3e}")
    print("== FFT closed form vs. direct sparse solve ==")
    for k, v in cf.items():
        print(f"  {k:28s} {v:.3e}")
    ok = (adj["div_adjoint_rel_err"] < 1e-12 and adj["sym_grad_adjoint_rel_err"] < 1e-12
          and max(cf.values()) < 1e-9)
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if report() else 1)
