"""Split-step evolution of PDEs on quantics tensor trains.

The equations this module solves are of the form

    d_t u(x, t) = l(-i grad) u  +  p(x) u  +  N(u, x, t),                       (1)

in any number of space dimensions, for a scalar or a multi-component field ``u``:

* ``l`` is a *Fourier multiplier* -- any linear, translation-invariant operator,
  specified by its symbol ``l(k)``. ``-i k**2/2`` is the free Schrodinger operator,
  ``-D k**2`` diffusion, ``-nu k**4`` hyperviscosity, ``-|k|**alpha`` a fractional
  Laplacian, ``i c k`` advection.
* ``p`` is a *local linear* term, a potential: ``-i V(x)`` for Schrodinger, ``-V(x)``
  in imaginary time, a spatially varying growth rate for a reaction-diffusion system.
* ``N`` is a *local nonlinearity*: at every point ``x`` it is an ODE in ``u`` alone.
  ``-i g |u|**2 u`` for Gross-Pitaevskii, ``r u (1 - u)`` for Fisher-KPP,
  ``u - u**3`` for Allen-Cahn, and any coupling between components of a
  multi-component field.

What (1) excludes is a nonlinear term containing derivatives (``u u_x`` in Burgers or
KdV) -- those are not local and the splitting below does not apply to them.

Method
------
Strang splitting. Over one step ``h`` the three parts are applied as

    N(h/2) P(h/2) L(h) P(h/2) N(h/2),                                             (2)

which is second order in ``h`` for any three operators. Consecutive ``N(h/2)`` factors
of neighbouring steps are merged into ``N(h)`` whenever nothing in between needs the
true state at a step boundary (a snapshot, a renormalisation), so a long run costs one
nonlinear application per step.

* ``L(h)`` is exact: quantics Fourier transform, multiply by ``exp(h l(k))`` as a
  diagonal MPO, transform back.
* ``P(h/2)`` is exact: a diagonal MPO of ``exp(h/2 p(x))``, built once by TCI.
* ``N(h)`` is applied by re-interpolating the mapped field: the current train is
  evaluated wherever TCI asks, the local ODE is advanced there -- by an exact flow map
  if the equation supplies one, by classical RK4 otherwise -- and the result is
  cross-interpolated into a new train. This is the step that cannot be written as a
  fixed operator, and it dominates the cost.

The same stepping loop drives :mod:`tnde.reference`, a dense-array implementation of
the identical scheme, so the two differ only by tensor-train truncation.

Conventions
-----------
The grid is periodic and the momentum ``k`` in ``l(k)`` is physical (see
:class:`tnde.grid.Grid`). Fields are complex128 throughout; a real equation simply
carries a zero imaginary part. Snapshot ``0`` is the initial state, snapshot ``j`` the
state at ``t0 + j dt``.
"""
from __future__ import annotations

import random as _pyrandom
import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from qutecipy import contract, crossinterpolate2
from qutecipy.globalpivot import DefaultGlobalPivotFinder, _OneSiteScan
from qutecipy.tensortrain.core import TensorTrain, reverse

from tnde import fit, tt
from tnde.batcheval import QuanticsBatchFunction, _pad, _tt_eval_numpy, _tt_eval_one
from tnde.fourier import fourier_mpo
from tnde.grid import Grid
from tnde.operators import DEFAULT_NSEARCHGLOBALPIVOT

CDTYPE = np.complex128


# ==========================================================================
# the equation
# ==========================================================================

def _per_component(obj, ncomp):
    """Normalise ``None`` / one callable / a sequence of callables to a list of
    length ``ncomp`` (``None`` entries meaning "absent for that component")."""
    if obj is None:
        return [None] * ncomp
    if callable(obj):
        return [obj] * ncomp
    obj = list(obj)
    if len(obj) != ncomp:
        raise ValueError(f"expected {ncomp} entries, got {len(obj)}")
    return obj


@dataclass
class Equation:
    """``d_t u = l(-i grad) u + p(x) u + N(u, x, t)``, see the module docstring.

    Parameters
    ----------
    linear
        Symbol ``l(k)`` (1D), ``l(kx, ky)`` (2D), ... returning ``d_t u_k / u_k``.
        For a multi-component field: one callable shared by every component, or a
        sequence with one entry per component (``None`` for none).
    potential
        ``p(x)``, ``p(x, y)``, ... returning ``d_t u / u``. Per component as above.
        Static; a time-dependent local term belongs in ``nonlinear``.
    nonlinear
        Right-hand side ``N(u, x, t)`` (1D), ``N(u, x, y, t)`` (2D), ... For
        ``ncomp > 1`` ``u`` is a tuple of arrays and a tuple is returned. Integrated
        with classical RK4 in ``rk_substeps`` sub-steps unless ``flow`` is given.
    flow
        Exact solution map of the local ODE: ``flow(u, x, t, h)`` returns ``u`` advanced
        from ``t`` to ``t + h`` (coordinates between ``u`` and ``t``, as for ``nonlinear``).
        Preferred whenever it is available -- it costs one evaluation instead of four
        and carries no time-stepping error.
    ncomp
        Number of field components.
    filter
        Optional momentum window ``w(k)`` multiplied into ``exp(h l(k))``. Needed for
        oscillatory propagators (Schrodinger type) on fine grids, where the phase
        ``exp(-i h k**2/2)`` at large ``k`` has no low-rank quantics representation;
        see :func:`tnde.equations.fermi_lowpass`.
    normalise
        ``True``: rescale to unit ``L2`` norm (summed over components) after every
        step -- the imaginary-time route to a ground state. ``"each"``: normalise
        every component separately, for a mixture whose populations are fixed.
    """
    linear: Callable | Sequence | None = None
    potential: Callable | Sequence | None = None
    nonlinear: Callable | None = None
    flow: Callable | None = None
    ncomp: int = 1
    filter: Callable | None = None
    normalise: bool | str = False
    rk_substeps: int = 1
    name: str = ""
    _linear: list = field(init=False, repr=False)
    _potential: list = field(init=False, repr=False)

    def __post_init__(self):
        if self.ncomp < 1:
            raise ValueError("ncomp must be >= 1")
        if self.nonlinear is not None and self.flow is not None:
            raise ValueError("give either `nonlinear` (RK4) or `flow` (exact), not both")
        self._linear = _per_component(self.linear, self.ncomp)
        self._potential = _per_component(self.potential, self.ncomp)

    @property
    def has_linear(self) -> bool:
        return any(s is not None for s in self._linear)

    @property
    def has_potential(self) -> bool:
        return any(p is not None for p in self._potential)

    @property
    def has_nonlinear(self) -> bool:
        return self.nonlinear is not None or self.flow is not None

    def symbol(self, c: int):
        return self._linear[c]

    def coefficient(self, c: int):
        return self._potential[c]

    # -- the local flow, shared by the tensor-train and the dense solvers --------

    def _wrap(self, u, like=None):
        us = [u] if self.ncomp == 1 else list(u)
        if like is None:
            return [np.asarray(a, dtype=CDTYPE) for a in us]
        return [np.broadcast_to(np.asarray(a, dtype=CDTYPE), like.shape).copy() for a in us]

    def _unwrap(self, u):
        return u[0] if self.ncomp == 1 else tuple(u)

    def local_flow(self, u: list, x: tuple, t: float, h: float) -> list:
        """Advance the local ODE ``du/dt = N(u, x, t)`` from ``t`` to ``t + h`` at a
        batch of points. ``u`` is a list of component arrays, ``x`` the coordinates."""
        if self.flow is not None:
            return self._wrap(self.flow(self._unwrap(u), *x, t, h), like=u[0])
        if self.nonlinear is None:
            return u
        rhs = self.nonlinear
        n = max(1, int(self.rk_substeps))
        hh = h / n

        def f(v, tv):
            return self._wrap(rhs(self._unwrap(v), *x, tv), like=u[0])

        for i in range(n):
            ti = t + i * hh
            k1 = f(u, ti)
            k2 = f([a + 0.5 * hh * b for a, b in zip(u, k1)], ti + 0.5 * hh)
            k3 = f([a + 0.5 * hh * b for a, b in zip(u, k2)], ti + 0.5 * hh)
            k4 = f([a + hh * b for a, b in zip(u, k3)], ti + hh)
            u = [a + hh / 6.0 * (b1 + 2 * b2 + 2 * b3 + b4)
                 for a, b1, b2, b3, b4 in zip(u, k1, k2, k3, k4)]
        return u


# ==========================================================================
# function -> train, function -> diagonal MPO
# ==========================================================================

def _bits_value(bits) -> int:
    return int("".join(str(int(b)) for b in bits), 2)


def peak_pivots(f: QuanticsBatchFunction, nsites: int, nrandom: int = 5, rng=None,
                extra=None) -> list[list[int]]:
    """Initial pivots at the peaks of ``|f|`` and their neighbours -- the batched
    counterpart of :func:`tnde.operators.peak_pivots`.

    From each of ``nrandom`` random starts, all ``nsites`` single-bit flips are
    evaluated in one call and the best is taken, until none improves. Seeding the
    neighbours ``i +- 1`` of every peak puts pivots in both branches of any bit
    boundary the peak sits on, which is what keeps TCI from interpolating one half
    of a centred function and reporting convergence.
    """
    rng = rng or np.random.default_rng(0)
    top = (1 << nsites) - 1
    flips = np.int64(1) << np.arange(nsites, dtype=np.int64)
    idxs = set(_bits_value(p) for p in (extra or []))
    for _ in range(nrandom):
        i = int(rng.integers(0, top, dtype=np.int64))
        val = abs(f.values(np.array([i]))[0])
        while True:
            cand = np.int64(i) ^ flips
            vals = np.abs(f.values(cand))
            b = int(np.argmax(vals))
            if vals[b] > val:
                i, val = int(cand[b]), float(vals[b])
            else:
                break
        idxs.update((i, max(i - 1, 0), min(i + 1, top)))
    if not idxs:
        idxs.add(0)
    return [[(i >> (nsites - 1 - s)) & 1 for s in range(nsites)] for i in sorted(idxs)]


class BatchedGlobalPivotFinder(DefaultGlobalPivotFinder):
    """TCI's global pivot search with the candidate points of each random start
    evaluated in one batch.

    The default finder scans every site of a start point and evaluates the function
    at each alternative value one call at a time -- ``2 * nsites`` scalar calls per
    start, and with ``nsearch = 20`` starts per sweep that is where most of a
    nonlinear step went. The candidates are known up front (the start and its
    ``nsites`` single-bit flips), so they are evaluated together through
    :meth:`QuanticsBatchFunction.values`; the search itself -- keep the largest
    error above ``abstol * tolmarginglobalsearch`` -- is unchanged.
    """

    def __call__(self, input, f, abstol, verbosity=0, rng=None):
        if not hasattr(f, "values"):
            return super().__call__(input, f, abstol, verbosity, rng)
        rng = rng if rng is not None else _pyrandom
        L = len(input.localdims)
        if any(d != 2 for d in input.localdims):
            return super().__call__(input, f, abstol, verbosity, rng)
        starts = [[rng.randrange(2) for _ in range(L)] for _ in range(self.nsearch)]
        starts += self._pivot_seeds(input)
        predict = _OneSiteScan(input.current_tt)
        flips = np.int64(1) << np.arange(L - 1, -1, -1, dtype=np.int64)   # site p <-> bit L-1-p

        found = []
        for point in starts:
            base = _bits_value(point)
            vals = f.values(np.concatenate([[base], np.int64(base) ^ flips]))
            best_error, best_point = 0.0, list(point)
            for p in range(L):
                pred = predict(point, p)
                e_base = abs(vals[0] - pred[point[p]])
                e_flip = abs(vals[1 + p] - pred[1 - point[p]])
                if e_base > best_error:
                    best_error, best_point = e_base, list(point)
                if e_flip > best_error:
                    best_error = e_flip
                    best_point = list(point)
                    best_point[p] = 1 - point[p]
            if best_error > abstol * self.tolmarginglobalsearch:
                found.append(tuple(best_point))
        if len(found) > self.maxnglobalpivot:
            found = found[: self.maxnglobalpivot]
        if verbosity > 0:
            print(f"Found {len(found)} global pivots")
        return found


def _vec(val, idx) -> np.ndarray:
    """A user function's return value as a complex array of the batch's shape -- a
    constant (``lambda x: 1.0``) broadcasts instead of failing."""
    return np.broadcast_to(np.asarray(val, dtype=CDTYPE), idx.shape).copy()


def _zero_train(nsites: int) -> TensorTrain:
    return TensorTrain([np.zeros((1, 2, 1), dtype=CDTYPE) for _ in range(nsites)])


def _tci(f: QuanticsBatchFunction, nsites: int, pivots, tolerance: float,
         maxdim: int | None) -> TensorTrain:
    """Run TCI on an evaluator, returning a zero train if the function vanishes on
    every seed pivot (TCI itself aborts on an identically zero function)."""
    if all(abs(f(p)) == 0.0 for p in pivots):
        return _zero_train(nsites)
    kw = {"tolerance": tolerance, "nsearchglobalpivot": DEFAULT_NSEARCHGLOBALPIVOT,
          "globalpivotfinder": BatchedGlobalPivotFinder(nsearch=DEFAULT_NSEARCHGLOBALPIVOT)}
    if maxdim is not None:
        kw["maxbonddim"] = maxdim
    ci, _, _ = crossinterpolate2(CDTYPE, f, [2] * nsites, pivots, **kw)
    return tt.from_cores(tt.cores(ci))


def interpolate(fn, grid: Grid, tolerance: float = 1e-10, maxdim: int | None = None,
                nrandom: int = 8, rng=None, pivots=None) -> TensorTrain:
    """Cross-interpolate ``fn(x, y, ...)`` into a quantics train on ``grid``.

    Seeds TCI with the points straddling the box centre plus ``nrandom`` peaks of
    ``|fn|`` found by greedy ascent from random starts (and any explicit ``pivots``).
    Check the result with :func:`max_error` -- TCI's own error estimate only sees the
    pivots it chose.
    """
    f = QuanticsBatchFunction(lambda idx: _vec(fn(*grid.coords(idx)), idx), grid.nsites)
    extra = grid.centre_pivots() + list(pivots or [])
    return _tci(f, grid.nsites, peak_pivots(f, grid.nsites, nrandom, rng, extra=extra),
                tolerance, maxdim)


def diagonal_mpo(fn, grid: Grid, tolerance: float = 1e-8, nrandom: int = 8, rng=None,
                 pivots=None) -> TensorTrain:
    """``fn(x, y, ...)`` as a diagonal MPO on ``grid``, so that contracting it with a
    train multiplies the field pointwise."""
    f = QuanticsBatchFunction(lambda idx: _vec(fn(*grid.coords(idx)), idx), grid.nsites)
    extra = grid.centre_pivots() + list(pivots or [])
    return tt.tt_to_mpo(_tci(f, grid.nsites,
                             peak_pivots(f, grid.nsites, nrandom, rng, extra=extra),
                             tolerance, None))


def multiplier_mpo(fn, grid: Grid, tolerance: float = 1e-8, nrandom: int = 8,
                   rng=None) -> TensorTrain:
    """``fn(kx, ky, ...)`` as a diagonal MPO on the *momentum* train, in the slot order
    :func:`fourier_transform` produces (variables reversed for ``ndim > 1``).

    The ``2**ndim`` corners of the index box are seeded as pivots: that is where the
    small momenta live, and a decaying multiplier such as ``exp(-h D k**2)`` is zero
    everywhere else.
    """
    f = QuanticsBatchFunction(
        lambda idx: _vec(fn(*grid.momenta(idx, reversed_slots=True)), idx), grid.nsites)
    return tt.tt_to_mpo(_tci(f, grid.nsites,
                             peak_pivots(f, grid.nsites, nrandom, rng,
                                         extra=grid.corner_pivots()),
                             tolerance, None))


# ==========================================================================
# Fourier transform of an interleaved train
# ==========================================================================

def fourier_transform(u: TensorTrain, grid: Grid, sign: float = -1.0,
                      tolerance: float = 1e-12, maxdim: int | None = None,
                      qft: TensorTrain | None = None,
                      fourier_tol: float = 1e-12) -> TensorTrain:
    """Discrete Fourier transform of every variable of an interleaved train.

    ``sign=-1`` is the forward transform (``numpy.fft`` convention), ``+1`` the
    inverse; both are unitary, so the pair composes to the identity.

    Output ordering: the train comes back MSB-first, but with the *variables in
    reverse order* -- slot ``s`` holds variable ``ndim - 1 - s``. Each per-variable QFT
    reverses its own digits in place and the final whole-train reversal that restores
    MSB-first order also flips the variable slots. In 1D nothing changes; in 2D the
    result is ordered ``(ky, kx)``. :func:`multiplier_mpo` builds its operator in this
    order, and a second transform (the inverse) flips the slots back.
    """
    R, d = grid.R, grid.ndim
    if len(u) != grid.nsites:
        raise ValueError(f"train has {len(u)} sites, grid has {grid.nsites}")
    if qft is None:
        qft = fourier_mpo(R, sign=sign, tolerance=fourier_tol)
    cores = [np.asarray(qft.sitetensor(i)) for i in range(R)]
    out = u
    for v in range(d):
        mpo = tt.embed_mpo(cores, range(v, d * R, d), d * R)
        out = contract(mpo, out, algorithm="naive", tolerance=tolerance, maxbonddim=maxdim)
    return reverse(out)


# ==========================================================================
# state utilities
# ==========================================================================

def _as_list(u):
    return list(u) if isinstance(u, (list, tuple)) else [u]


def _as_state(comps, ncomp):
    return comps[0] if ncomp == 1 else list(comps)


def norm(u, grid: Grid) -> float:
    """``sqrt(sum_c int |u_c|**2 dV)``."""
    return float(np.sqrt(sum(tt.inner(c, c).real for c in _as_list(u)) * grid.dvol))


def normalise(u, grid: Grid, each: bool = False):
    """Rescale to unit :func:`norm`, spreading the factor evenly over the cores.
    ``each=True`` normalises every component separately instead of their sum."""
    comps = _as_list(u)
    norms = [norm(c, grid) for c in comps] if each else [norm(comps, grid)] * len(comps)
    if any(n == 0.0 for n in norms):
        raise ZeroDivisionError("cannot normalise a zero field")
    out = [tt.from_cores([c * n ** (-1.0 / grid.nsites) for c in tt.cores(x)])
           for x, n in zip(comps, norms)]
    return _as_state(out, len(comps))


def integral(u: TensorTrain, grid: Grid) -> complex:
    """``int u dV`` -- the sum of every entry of the train times the volume element."""
    ones = TensorTrain([np.ones((1, 2, 1), dtype=CDTYPE) for _ in range(grid.nsites)])
    return tt.inner(ones, u) * grid.dvol


def evaluate(u: TensorTrain, grid: Grid, *x):
    """The field at the grid points nearest to real coordinates ``x[v]`` -- a scalar
    for scalar coordinates, an array of their broadcast shape otherwise."""
    idx = grid.nearest(*x)
    out = tt.evaluate(u, idx.ravel()).reshape(idx.shape)
    return out[()] if out.ndim == 0 else out


def to_dense(u: TensorTrain, grid: Grid) -> np.ndarray:
    """The full ``(M,) * ndim`` array. Small grids only."""
    return tt.evaluate(u, grid.dense_indices().ravel()).reshape((grid.M,) * grid.ndim)


def reconstruct(u: TensorTrain, grid: Grid, n: int = 256, window=None):
    """The field on an ``n``-per-direction sample of the box (or of ``window``).
    Returns ``(axes, values)`` with ``values`` of shape ``(n,) * ndim``."""
    idx, axes = grid.sample(n, window)
    return axes, tt.evaluate(u, idx.ravel()).reshape(idx.shape)


def max_error(u: TensorTrain, fn, grid: Grid, n: int = 256, window=None) -> float:
    """Maximum relative deviation of a train from ``fn`` on an independent sample."""
    idx, _ = grid.sample(n, window)
    exact = np.asarray(fn(*grid.coords(idx.ravel())), dtype=CDTYPE)
    got = tt.evaluate(u, idx.ravel())
    scale = np.max(np.abs(exact))
    return float(np.max(np.abs(got - exact)) / scale) if scale > 0 else float(np.max(np.abs(got)))


# ==========================================================================
# the split-step loop, shared with the dense reference
# ==========================================================================

def run_splitting(step, u, dt: float, nsteps: int, t0: float = 0.0,
                  save_every: int | None = None, normalise: bool = False,
                  callback=None, log=None):
    """Drive the merged Strang sequence (2) with the operations of ``step``.

    ``step`` provides ``linear(u)`` for ``L(dt)``, ``potential(u)`` for ``P(dt/2)``,
    ``nonlinear(u, t, h)`` for ``N(h)`` starting at time ``t``, ``normalise(u)`` and
    ``describe(u)`` for logging. The half-steps of consecutive nonlinear applications
    are merged except where the true state at a step boundary is needed.
    """
    snapshots = {0: u}
    t = t0
    if callback is not None:
        callback(0, t, u)
    half = dt / 2
    u = step.potential(step.nonlinear(u, t, half))
    for j in range(1, nsteps + 1):
        u = step.potential(step.linear(u))
        boundary = normalise or j == nsteps or (save_every and j % save_every == 0)
        if boundary:
            u = step.nonlinear(u, t + half, half)
            t = t0 + j * dt
            if normalise:
                u = step.normalise(u)
            if (save_every and j % save_every == 0) or j == nsteps:
                snapshots[j] = u
                if log:
                    log(f"  step {j:>6}/{nsteps}  t = {t:<10.5g} {step.describe(u)}")
                if callback is not None:
                    callback(j, t, u)
            if j < nsteps:
                u = step.potential(step.nonlinear(u, t, half))
        else:
            u = step.potential(step.nonlinear(u, t + half, dt))
            t = t0 + j * dt
    return u, snapshots


class TrainStepper:
    """The three split-step operations on quantics trains, with everything that can
    be precomputed done once: the ``P(dt/2)`` MPOs, the ``exp(dt l(k))`` multiplier
    MPOs and the two Fourier MPOs."""

    def __init__(self, eq: Equation, grid: Grid, dt: float, tolerance: float = 1e-10,
                 maxdim: int = 32, operator_tol: float | None = None,
                 method: str = "auto", nsweeps: int = 2, fourier_tol: float = 1e-12,
                 nrandom: int = 5, rng=None, log=None):
        self.eq, self.grid, self.dt = eq, grid, dt
        self.tol = tolerance
        self.maxdim = maxdim
        if method == "auto":
            # In 1D the diagonal MPOs have small rank and the exact product is cheap
            # and slightly more accurate. In 2D and 3D a separable function has
            # *multiplicative* rank on the interleaved train -- a Gaussian times a
            # Gaussian is rank r_x * r_y at the cross bonds -- so both the state and the
            # momentum multiplier sit at rank 50-70 and the exact product's SVDs cost
            # ~50x the variational fit (measured: 57 s against 1 s per application at
            # R = 10).
            method = "naive" if grid.ndim == 1 else "fit"
        self.method, self.nsweeps = method, nsweeps
        self.nrandom = nrandom
        self.rng = rng or np.random.default_rng(0)
        optol = tolerance if operator_tol is None else operator_tol
        t0 = time.time()

        self.pot = []
        for c in range(eq.ncomp):
            p = eq.coefficient(c)
            self.pot.append(None if p is None else
                            diagonal_mpo(lambda *x, p=p: np.exp(0.5 * dt * p(*x)), grid, optol,
                                         rng=self.rng))
        if eq.has_potential and log:
            log(f"  potential MPOs: ranks {[o.rank() for o in self.pot if o is not None]}"
                f"  ({time.time()-t0:.1f}s)")

        self.mult = []
        if eq.has_linear:
            self.qft = fourier_mpo(grid.R, sign=-1.0, tolerance=fourier_tol)
            self.iqft = fourier_mpo(grid.R, sign=+1.0, tolerance=fourier_tol)
        for c in range(eq.ncomp):
            s = eq.symbol(c)
            if s is None:
                self.mult.append(None)
                continue
            w = eq.filter

            def prop(*k, s=s, w=w):
                out = np.exp(dt * np.asarray(s(*k), dtype=CDTYPE))
                return out if w is None else out * w(*k)
            self.mult.append(multiplier_mpo(prop, grid, optol, rng=self.rng))
        if eq.has_linear and log:
            log(f"  momentum MPOs: ranks {[o.rank() for o in self.mult if o is not None]}"
                f"  ({time.time()-t0:.1f}s)")

    # -- operator application ---------------------------------------------------

    def _apply(self, mpo, u):
        if self.method == "fit":
            return fit.apply_mpo(mpo, u, self.maxdim, nsweeps=self.nsweeps, tolerance=self.tol)
        return contract(mpo, u, algorithm="naive", tolerance=self.tol, maxbonddim=self.maxdim)

    def linear(self, u):
        if not self.eq.has_linear:
            return u
        out = []
        for c, x in enumerate(_as_list(u)):
            m = self.mult[c]
            if m is None:
                out.append(x)
                continue
            # The Fourier MPOs are non-local, so they always go through the exact
            # product: a two-sweep fit from the state as a guess converges poorly.
            k = fourier_transform(x, self.grid, -1.0, self.tol, self.maxdim, qft=self.qft)
            k = self._apply(m, k)
            out.append(fourier_transform(k, self.grid, +1.0, self.tol, self.maxdim, qft=self.iqft))
        return _as_state(out, self.eq.ncomp)

    def potential(self, u):
        if not self.eq.has_potential:
            return u
        out = [x if self.pot[c] is None else self._apply(self.pot[c], x)
               for c, x in enumerate(_as_list(u))]
        return _as_state(out, self.eq.ncomp)

    def nonlinear(self, u, t, h):
        """Re-interpolate ``flow_h(u)``: evaluate every component wherever TCI asks,
        advance the local ODE there, and cross-interpolate each output component.

        The mapped values are cached per grid index and shared between the
        components' interpolations, so the flow is evaluated once per point.
        """
        if not self.eq.has_nonlinear:
            return u
        comps = _as_list(u)
        grid, eq, nsites = self.grid, self.eq, self.grid.nsites
        cores = [tt.cores(c) for c in comps]
        D = max(max(max(c.shape[0], c.shape[2]) for c in cs) for cs in cores)
        pads = [_pad(cs, D) for cs in cores]
        cache = {}

        def mapped(idx):
            idx = np.asarray(idx, dtype=np.int64)
            if idx.size == 1:                       # TCI's pivot searches come one point at a time
                i = int(idx[0])
                v = cache.get(i)
                if v is None:
                    vals = [np.array([_tt_eval_one(p, i, nsites)]) for p in pads]
                    x = tuple(np.array([c]) for c in grid.coord_of(i))
                    v = np.array([a[0] for a in eq.local_flow(vals, x, t, h)])
                    cache[i] = v
                return v[None, :]
            uniq, inv = np.unique(idx, return_inverse=True)
            out = np.empty((uniq.size, eq.ncomp), dtype=CDTYPE)
            missing = np.array([cache.get(i) is None for i in uniq.tolist()])
            for n, i in enumerate(uniq.tolist()):
                if not missing[n]:
                    out[n] = cache[i]
            if missing.any():
                todo = uniq[missing]
                vals = [_tt_eval_numpy(p, todo, nsites, D) for p in pads]
                new = np.stack(eq.local_flow(vals, grid.coords(todo), t, h), axis=1)
                out[missing] = new
                cache.update(zip(todo.tolist(), new))
            return out[inv]

        out = []
        for c in range(eq.ncomp):
            f = QuanticsBatchFunction(lambda idx, c=c: mapped(idx)[:, c], nsites, cache=False)
            out.append(_tci(f, nsites, peak_pivots(f, nsites, self.nrandom, self.rng),
                            self.tol, self.maxdim))
        return _as_state(out, eq.ncomp)

    def normalise(self, u):
        return normalise(u, self.grid, each=self.eq.normalise == "each")

    def describe(self, u) -> str:
        comps = _as_list(u)
        ranks = [c.rank() for c in comps]
        return f"rank {ranks if len(ranks) > 1 else ranks[0]!s:<10} norm {norm(comps, self.grid):.9f}"


def initial_state(u0, eq: Equation, grid: Grid, tolerance: float = 1e-10,
                  maxdim: int | None = None, rng=None):
    """Turn ``u0`` -- a callable, a train, or one per component -- into a state."""
    comps = _as_list(u0) if (eq.ncomp > 1 or isinstance(u0, (list, tuple))) else [u0]
    if len(comps) != eq.ncomp:
        raise ValueError(f"expected {eq.ncomp} initial components, got {len(comps)}")
    out = []
    for c in comps:
        if callable(c):
            out.append(interpolate(c, grid, tolerance, maxdim, rng=rng))
        elif len(c) == grid.nsites:
            out.append(c)
        else:
            raise ValueError(f"initial train has {len(c)} sites, grid has {grid.nsites}")
    return _as_state(out, eq.ncomp)


def solve(eq: Equation, u0, grid: Grid, dt: float, nsteps: int, tolerance: float = 1e-10,
          maxdim: int = 32, operator_tol: float | None = None, method: str = "auto",
          nsweeps: int = 2, fourier_tol: float = 1e-12, t0: float = 0.0,
          save_every: int | None = None, callback=None, nrandom: int = 5, rng=None,
          verbose: bool = True):
    """Evolve ``u0`` under ``eq`` on ``grid`` for ``nsteps`` steps of ``dt``.

    Returns ``(u, snapshots)``: the final state and a dict ``{step: state}`` holding the
    initial state under ``0``, every ``save_every``-th step, and the last. A state is
    a ``TensorTrain`` for a scalar equation and a list of them otherwise.

    ``tolerance`` and ``maxdim`` bound every cross interpolation and every truncation
    of the field; ``operator_tol`` (default ``tolerance``) is the TCI tolerance of the
    fixed operators. ``method`` chooses how the diagonal MPOs are applied: ``'naive'``
    forms the exact product and truncates it, ``'fit'`` sweeps a variational ansatz
    of bond dimension ``maxdim`` -- much faster when the ranks are large, slightly
    less accurate when they are not. ``'auto'`` (the default) is ``'naive'`` in 1D and
    ``'fit'`` in 2D and 3D, where ranks are inherently larger. The Fourier MPOs always
    go through the exact product. ``callback(step, t, state)`` is called at every
    snapshot.
    """
    log = print if verbose else None
    t_start = time.time()
    u = initial_state(u0, eq, grid, tolerance, maxdim, rng=rng)
    if eq.normalise:
        u = normalise(u, grid, each=eq.normalise == "each")
    step = TrainStepper(eq, grid, dt, tolerance, maxdim, operator_tol, method, nsweeps,
                        fourier_tol, nrandom, rng, log)
    if log:
        log(f"  initial state: {step.describe(u)}  ({time.time()-t_start:.1f}s)")
    u, snaps = run_splitting(step, u, dt, nsteps, t0, save_every, eq.normalise, callback, log)
    if log:
        log(f"  done in {time.time()-t_start:.1f}s")
    return u, snaps
