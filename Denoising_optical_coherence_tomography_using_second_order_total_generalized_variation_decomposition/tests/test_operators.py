"""
Unit tests for the discrete operators, the closed-form split-Bregman updates and the
TGV -> TV limit.

Run with:   .venv/bin/python -m pytest tests/ -q
       or:  .venv/bin/python tests/test_operators.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import verify                                                  # noqa: E402
from src.operators import (div, div_sym, grad, isotropic_shrink,        # noqa: E402
                           sym_grad, sym_grad_adjoint)
from src.tgv import tgv_decompose, vo_decompose                         # noqa: E402


def test_div_is_negative_adjoint_of_grad():
    r = verify.check_adjoints(shape=(9, 13))
    assert r["div_adjoint_rel_err"] < 1e-12
    assert r["max_abs_div_minus_negGt"] < 1e-12


def test_sym_grad_and_its_adjoint():
    r = verify.check_adjoints(shape=(9, 13))
    assert r["sym_grad_adjoint_rel_err"] < 1e-12


def test_closed_form_updates_match_direct_solve():
    r = verify.check_closed_forms(shape=(8, 11))
    assert max(r.values()) < 1e-9, r


def test_isotropic_shrink_is_the_proximal_map():
    """shrink(z, t) must minimise 0.5||y-z||^2 + t||y||_2.

    Checked through the exact optimality conditions of the prox:
        y != 0  =>  y - z + t*y/|y| = 0
        y == 0  =>  |z| <= t
    plus a direct comparison of objective values against a dense 1-D search along the
    segment [0, z] (the solution necessarily lies there).
    """
    rng = np.random.default_rng(0)
    z1, z2 = rng.standard_normal((3, 4)), rng.standard_normal((3, 4))
    for t in (0.0, 0.15, 1.0):
        y1, y2 = isotropic_shrink((z1, z2), t)
        mag = np.hypot(y1, y2)
        # optimality conditions
        nz = mag > 1e-12
        res = np.hypot(y1[nz] - z1[nz] + t * y1[nz] / mag[nz],
                       y2[nz] - z2[nz] + t * y2[nz] / mag[nz])
        assert np.all(res < 1e-10), res.max()
        zz = np.hypot(z1, z2)
        assert np.all(zz[~nz] <= t + 1e-12)
        # objective value vs. a dense search along the ray from 0 to z
        for i in range(3):
            for j in range(4):
                z = np.array([z1[i, j], z2[i, j]])
                zm = np.linalg.norm(z)
                if zm == 0:
                    continue
                ss = np.linspace(0, zm, 2001)
                objs = [0.5 * np.sum((s * z / zm - z) ** 2) + t * s for s in ss]
                best = ss[int(np.argmin(objs))]
                y = np.array([y1[i, j], y2[i, j]])
                assert abs(np.linalg.norm(y) - best) < 1e-3 + 0.01 * zm, (t, best, y)


def test_shrink_handles_zero_division():
    y1, y2 = isotropic_shrink((np.zeros((4, 4)), np.zeros((4, 4))), 0.3)
    assert np.all(np.isfinite(y1)) and np.allclose(y1, 0.0)
    assert np.all(np.isfinite(y2)) and np.allclose(y2, 0.0)


def test_texture_is_f_minus_u():
    rng = np.random.default_rng(1)
    f = rng.random((32, 32))
    u, tex = tgv_decompose(f, 0.1, 0.1, 0.1, 2.0, 2.0, 2.0, max_iter=10)
    assert np.allclose(tex, f - u)


def test_tgv_reduces_to_vo_for_large_alpha0():
    """With alpha0 -> infinity the TGV term becomes TV, so TGV must reproduce VO."""
    rng = np.random.default_rng(2)
    f = rng.random((48, 48)) * 0.5
    uv, _ = vo_decompose(f, alpha1=0.1, beta=0.1, lam1=2.0, lam2=2.0,
                         max_iter=400, tol=1e-6)
    ut, _ = tgv_decompose(f, alpha0=100.0, alpha1=0.1, beta=0.1,
                          lam1=2.0, lam2=2.0, lam3=2.0, max_iter=400, tol=1e-6)
    assert np.abs(ut - uv).max() < 0.05 * (f.max() - f.min())


def test_energy_decreases_and_stopping_works():
    rng = np.random.default_rng(3)
    f = np.zeros((64, 64))
    f[16:48, 16:48] = 1.0
    f += 0.1 * rng.standard_normal(f.shape)
    u, tex, info = tgv_decompose(f, 0.1, 0.1, 0.1, 2.0, 2.0, 2.0,
                                 max_iter=300, tol=1e-3, return_info=True)
    assert info["n_iter"] < 300
    assert info["rel_change"][-1] < 1e-3
    # the *model* objective must be non-increasing up to numerical noise
    assert info["energy_model"][-1] <= info["energy_model"][0] + 1e-9
    assert np.all(np.isfinite(u))


if __name__ == "__main__":
    import traceback
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception:
                fails += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    raise SystemExit(1 if fails else 0)
