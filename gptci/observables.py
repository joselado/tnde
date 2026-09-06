"""Observables and wave-function reconstruction (``observables_1D.jl``).

A warning that applies to everything here: a quantics MPO for ``x`` or ``x**2`` is
built by TCI to a tolerance that is *relative to the function's maximum*. On the
paper's box that maximum is ``500**2 = 2.5e5``, so a tolerance of ``1e-6`` buys an
absolute accuracy of ~0.25 against an expectation value of order 0.5. That is why the
committed ``widths.txt`` starts at 0.8117 when the analytic answer is
``sqrt(1/2) = 0.7071``. Use a tolerance small enough for the box, or compute the
moments by direct summation with :func:`width_dense`.
"""
from __future__ import annotations

import numpy as np
from qutecipy import contract

from gptci import tt
from gptci.operators import pos_mpo, pos_squared_mpo


def expectation_value(psi, op, R, xmin, xmax, tolerance=1e-10) -> float:
    """``Re <psi| op |psi> dx`` with ``dx = L/2**R`` (``expectation_value_ITensor``)."""
    op_psi = contract(op, psi, algorithm="naive", tolerance=tolerance)
    dx = (xmax - xmin) / (1 << R)
    return float(np.real(tt.inner(psi, op_psi) * dx))


def width(psi, R, xmin, xmax, tolerance=1e-10, mpos=None) -> float:
    """``sqrt(<x**2> - <x>**2)`` via position MPOs.

    Pass ``mpos=(pos, pos_squared)`` to reuse the operators across many snapshots --
    building them is far more expensive than using them.
    """
    if mpos is None:
        mpos = (pos_mpo(R, xmin, xmax, tolerance), pos_squared_mpo(R, xmin, xmax, tolerance))
    x1 = expectation_value(psi, mpos[0], R, xmin, xmax, tolerance)
    x2 = expectation_value(psi, mpos[1], R, xmin, xmax, tolerance)
    return float(np.sqrt(max(x2 - x1**2, 0.0)))


def width_dense(psi, R, xmin, xmax, npoints=2**16, halfwidth=None) -> float:
    """``sqrt(<x**2> - <x>**2)`` by direct summation over a reconstructed sample.

    Free of the MPO tolerance problem above, and exact to the sampling density. Prefer
    this unless you specifically want to reproduce the original's numbers.
    """
    idx, x = _sample(R, xmin, xmax, npoints, halfwidth)
    rho = np.abs(tt.evaluate(psi, idx)) ** 2
    w = rho / rho.sum()
    x1 = float(np.sum(w * x))
    x2 = float(np.sum(w * x**2))
    return float(np.sqrt(max(x2 - x1**2, 0.0)))


def _sample(R, xmin, xmax, npoints, halfwidth=None, centre=None):
    dx = (xmax - xmin) / ((1 << R) - 1)
    if halfwidth is None:
        idx = np.round(np.linspace(0, (1 << R) - 1, npoints)).astype(np.int64)
    else:
        c = (1 << (R - 1)) if centre is None else int(round((centre - xmin) / dx))
        w = int(halfwidth / dx)
        idx = np.clip(c + np.round(np.linspace(-w, w, npoints)).astype(np.int64),
                      0, (1 << R) - 1)
    return idx, xmin + idx * dx


def reconstruct(psi, R, xmin, xmax, prec=12):
    """The wave function on ``2**prec`` points spanning the whole box
    (``evaluate_wavefunction``). Returns ``(x, psi(x))``."""
    idx, x = _sample(R, xmin, xmax, 1 << prec)
    return x, tt.evaluate(psi, idx)


def reconstruct_window(psi, R, xmin, xmax, x0, halfwidth, prec=12):
    """The wave function on ``2**prec`` points in ``[x0-halfwidth, x0+halfwidth]``
    (``evaluate_wavefunction_reduced``) -- the zoomed heatmaps in the paper."""
    idx, x = _sample(R, xmin, xmax, 1 << prec, halfwidth=halfwidth, centre=x0)
    return x, tt.evaluate(psi, idx)


def heatmap(snapshots, R, xmin, xmax, prec=12, window=None):
    """``|psi|`` over (time, space) for a dict of snapshots.

    Returns ``(steps, x, values)`` with ``values`` of shape ``(nsnapshots, 2**prec)``.
    ``window=(x0, halfwidth)`` zooms in, as ``observables_1D.jl`` does for its
    reduced heatmaps.
    """
    steps = sorted(snapshots)
    rows = []
    for s in steps:
        if window is None:
            x, v = reconstruct(snapshots[s], R, xmin, xmax, prec)
        else:
            x, v = reconstruct_window(snapshots[s], R, xmin, xmax, *window, prec=prec)
        rows.append(v)
    return np.asarray(steps), x, np.asarray(rows)


# --------------------------------------------------------------------------
# 2D
# --------------------------------------------------------------------------

def reconstruct_2d(psi, R, xmin, xmax, ymin, ymax, prec=8, window=None):
    """``psi(x, y)`` on a ``2**prec x 2**prec`` sample of an interleaved 2D train
    (``evaluate_2D``). Returns ``(x, y, values)``.

    ``window=(x0, y0, halfwidth)`` zooms in. Note the 2D grid has no
    ``includeendpoint``, so its spacing is ``L/2**R``.
    """
    from gptci import batcheval

    n = 1 << prec
    M = 1 << R
    dx, dy = (xmax - xmin) / M, (ymax - ymin) / M
    if window is None:
        gx = np.round(np.linspace(0, M - 1, n)).astype(np.int64)
        gy = np.round(np.linspace(0, M - 1, n)).astype(np.int64)
    else:
        x0, y0, hw = window
        cx, cy = int(round((x0 - xmin) / dx)), int(round((y0 - ymin) / dy))
        gx = np.clip(cx + np.round(np.linspace(-hw / dx, hw / dx, n)).astype(np.int64), 0, M - 1)
        gy = np.clip(cy + np.round(np.linspace(-hw / dy, hw / dy, n)).astype(np.int64), 0, M - 1)
    IX, IY = np.meshgrid(gx, gy, indexing="ij")
    vals = tt.evaluate(psi, batcheval.interleave(IX.ravel(), IY.ravel(), R)).reshape(n, n)
    return xmin + gx * dx, ymin + gy * dy, vals


def density_2d(psi, R, xmin, xmax, ymin, ymax, prec=8, window=None):
    """``|psi(x,y)|**2`` on a sample -- the quantity the paper's 2D figures show."""
    x, y, v = reconstruct_2d(psi, R, xmin, xmax, ymin, ymax, prec, window)
    return x, y, np.abs(v) ** 2


def _moment_mpo_2d(f, R, xmin, xmax, ymin, ymax, tolerance):
    from qutecipy import crossinterpolate2
    from gptci import batcheval
    from gptci.operators import DEFAULT_NSEARCHGLOBALPIVOT, peak_pivots

    fn = batcheval.grid_function_2d(lambda x, y: f(x, y).astype(np.complex128),
                                    R, xmin, xmax, ymin, ymax)
    ci, _, _ = crossinterpolate2(np.complex128, fn, [2] * (2 * R),
                                 peak_pivots(fn, 2 * R, 8),
                                 tolerance=tolerance,
                                 nsearchglobalpivot=DEFAULT_NSEARCHGLOBALPIVOT)
    return tt.tt_to_mpo(ci)


def moment_mpos_2d(R, xmin, xmax, ymin, ymax, tolerance=1e-10):
    """The position MPOs of ``observables_2D.jl``: x, x**2, y, y**2, r, r**2.

    Building them is far more expensive than using them, so build once and pass the
    dict to :func:`expectation_value_2d` for every snapshot.

    Same tolerance caveat as in 1D, squared: TCI's tolerance is relative to the
    function's maximum, which for ``x**2`` on a box of half-width 100 is 1e4.
    """
    r2 = lambda x, y: x**2 + y**2
    return {
        "x": _moment_mpo_2d(lambda x, y: x, R, xmin, xmax, ymin, ymax, tolerance),
        "x2": _moment_mpo_2d(lambda x, y: x**2, R, xmin, xmax, ymin, ymax, tolerance),
        "y": _moment_mpo_2d(lambda x, y: y, R, xmin, xmax, ymin, ymax, tolerance),
        "y2": _moment_mpo_2d(lambda x, y: y**2, R, xmin, xmax, ymin, ymax, tolerance),
        "r": _moment_mpo_2d(lambda x, y: np.sqrt(r2(x, y)), R, xmin, xmax, ymin, ymax, tolerance),
        "r2": _moment_mpo_2d(r2, R, xmin, xmax, ymin, ymax, tolerance),
    }


def expectation_value_2d(psi, op, R, xmin, xmax, ymin, ymax, tolerance=1e-10) -> float:
    """``Re <psi| op |psi> dx dy`` (``expectation_value_ITensor_2D``)."""
    from qutecipy import contract

    op_psi = contract(op, psi, algorithm="naive", tolerance=tolerance)
    dxdy = (xmax - xmin) / (1 << R) * (ymax - ymin) / (1 << R)
    return float(np.real(tt.inner(psi, op_psi) * dxdy))


def moments_2d(psi, R, xmin, xmax, ymin, ymax, mpos=None, tolerance=1e-10) -> dict:
    """``<x>, <y>, <r>, <r**2>`` and the widths in x and y, as ``observables_2D.jl``
    computes them."""
    if mpos is None:
        mpos = moment_mpos_2d(R, xmin, xmax, ymin, ymax, tolerance)
    ev = {k: expectation_value_2d(psi, o, R, xmin, xmax, ymin, ymax, tolerance)
          for k, o in mpos.items()}
    return {
        "x": ev["x"], "y": ev["y"], "r": ev["r"], "r2": ev["r2"],
        "width_x": float(np.sqrt(max(ev["x2"] - ev["x"] ** 2, 0.0))),
        "width_y": float(np.sqrt(max(ev["y2"] - ev["y"] ** 2, 0.0))),
    }


def moments_2d_dense(psi, R, xmin, xmax, ymin, ymax, prec=8, window=None) -> dict:
    """The same moments by direct summation over a reconstructed sample.

    Free of the MPO tolerance problem, and the reference the MPO route is gated
    against. Prefer it unless you specifically want the original's numbers.
    """
    x, y, v = reconstruct_2d(psi, R, xmin, xmax, ymin, ymax, prec, window)
    X, Y = np.meshgrid(x, y, indexing="ij")
    w = np.abs(v) ** 2
    w = w / w.sum()
    ex, ey = float(np.sum(w * X)), float(np.sum(w * Y))
    r = np.sqrt(X**2 + Y**2)
    return {
        "x": ex, "y": ey,
        "r": float(np.sum(w * r)), "r2": float(np.sum(w * r**2)),
        "width_x": float(np.sqrt(max(float(np.sum(w * X**2)) - ex**2, 0.0))),
        "width_y": float(np.sqrt(max(float(np.sum(w * Y**2)) - ey**2, 0.0))),
    }
