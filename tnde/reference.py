"""Dense-array reference for :mod:`tnde.pde`.

The same :class:`~tnde.pde.Equation`, the same :class:`~tnde.grid.Grid`, the same
split-step loop (:func:`tnde.pde.run_splitting`) and the same local flow
(:meth:`Equation.local_flow`) -- only the state is a full ``(M,) * ndim`` array and the
Fourier multiplier goes through ``numpy.fft``. Every difference between this and the
tensor-train solver is therefore truncation, which is what the tests need to isolate.

Usable up to a few million grid points; the tensor-train path exists for the rest.
"""
from __future__ import annotations

import numpy as np

from tnde.grid import Grid
from tnde.pde import CDTYPE, Equation, run_splitting

_as_list = lambda u: list(u) if isinstance(u, (list, tuple)) else [u]


class DenseStepper:
    def __init__(self, eq: Equation, grid: Grid, dt: float):
        self.eq, self.grid, self.dt = eq, grid, dt
        self.X = grid.mesh()
        K = grid.kmesh()
        self.mult, self.pot = [], []
        for c in range(eq.ncomp):
            s, p = eq.symbol(c), eq.coefficient(c)
            if s is None:
                self.mult.append(None)
            else:
                m = np.exp(dt * np.asarray(s(*K), dtype=CDTYPE))
                self.mult.append(m if eq.filter is None else m * eq.filter(*K))
            self.pot.append(None if p is None else np.exp(0.5 * dt * np.asarray(p(*self.X), dtype=CDTYPE)))

    def _state(self, comps):
        return comps[0] if self.eq.ncomp == 1 else comps

    def linear(self, u):
        return self._state([x if m is None else np.fft.ifftn(m * np.fft.fftn(x))
                            for x, m in zip(_as_list(u), self.mult)])

    def potential(self, u):
        return self._state([x if p is None else p * x for x, p in zip(_as_list(u), self.pot)])

    def nonlinear(self, u, t, h):
        if not self.eq.has_nonlinear:
            return u
        return self._state(self.eq.local_flow(_as_list(u), self.X, t, h))

    def normalise(self, u):
        comps = _as_list(u)
        if self.eq.normalise == "each":
            return self._state([x / np.sqrt(np.sum(np.abs(x) ** 2) * self.grid.dvol) for x in comps])
        n = np.sqrt(sum(np.sum(np.abs(x) ** 2) for x in comps) * self.grid.dvol)
        return self._state([x / n for x in comps])

    def describe(self, u) -> str:
        comps = _as_list(u)
        n = np.sqrt(sum(np.sum(np.abs(x) ** 2) for x in comps) * self.grid.dvol)
        return f"norm {n:.9f}"


def solve(eq: Equation, u0, grid: Grid, dt: float, nsteps: int, t0: float = 0.0,
          save_every: int | None = None, callback=None, verbose: bool = False):
    """Dense counterpart of :func:`tnde.pde.solve`; ``u0`` is a callable on the
    coordinates or an array (one per component for ``ncomp > 1``)."""
    X = grid.mesh()
    comps = _as_list(u0) if (eq.ncomp > 1 or isinstance(u0, (list, tuple))) else [u0]
    comps = [np.broadcast_to(np.asarray(c(*X) if callable(c) else c, dtype=CDTYPE), X[0].shape).copy()
             for c in comps]
    step = DenseStepper(eq, grid, dt)
    u = step._state(comps)
    if eq.normalise:
        u = step.normalise(u)
    return run_splitting(step, u, dt, nsteps, t0, save_every, eq.normalise, callback,
                         print if verbose else None)
