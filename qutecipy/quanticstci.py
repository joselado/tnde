"""Quantics tensor cross interpolation: the two halves of this package, joined.

**This module is an extension: it has no vendored Julia counterpart.** Every other
module here ports a specific ``.jl`` file, and ``reference/`` holds
``TensorCrossInterpolation.jl``, ``QuadGK.jl`` and ``QuanticsGrids.jl`` -- but *not*
``QuanticsTCI.jl``. So this is our own design, written in the spirit of the rest of
the package rather than translated from it, and it makes no claim of API parity with
the Julia package of a similar name. It is validated against direct
``crossinterpolate2``-on-a-quantics-grid runs (which it must reproduce bit for bit)
and against closed-form integrals, not against live Julia output.

What it is for
--------------
``qutecipy.quantics`` maps a coordinate ``x`` to a string of base-``b`` digits, and
``qutecipy.tci2`` cross-interpolates functions of many small indices. Composing them
is what gives exponential compression for functions with scale separation -- a
2^20-point grid represented at bond dimension 3 -- but composing them by hand means
building the grid, wrapping ``f``, calling ``crossinterpolate2`` with
``localdimensions()``, and then converting coordinates to quantics indices on *every*
evaluation and remembering the step-size factor on *every* integral::

    qf = quantics_function(np.float64, grid, f)
    tci, _, _ = crossinterpolate2(np.float64, qf, grid.localdimensions(), tolerance=1e-9)
    tci.evaluate(list(grid.origcoord_to_quantics(x)))              # every evaluation
    TensorTrain(tci.sitetensors()).sum() * grid.grid_step()[0]     # every integral

:func:`quantics_crossinterpolate` does that once and hands back a
:class:`QuanticsTensorCI` you can call with ordinary coordinates.

Caching is on by default here
-----------------------------
Bare ``crossinterpolate2`` does *not* cache, matching the Julia original, because a
cache hit only pays when ``f`` costs more than a dict-of-tuple lookup (see the
performance notes in CLAUDE.md -- Python tuples do not cache their hash, so that
floor is real). A *quantics* ``f`` is never that cheap: every call decodes a
mixed-radix digit string into a coordinate before the user's function even runs.
Measured on the R=20 example below, caching cut the run from 119 ms to 45 ms
(**2.6x**) and the number of calls to the user's ``f`` from 8846 to 1502. So
``cache=True`` is the default here, deliberately diverging from
``crossinterpolate2``; pass ``cache=False`` to match it.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from qutecipy.quantics.discretized import DiscretizedGrid, quantics_function
from qutecipy.tci2 import crossinterpolate2
from qutecipy.tensortrain.cachedfunction import CachedFunction
from qutecipy.tensortrain.core import TensorTrain


class QuanticsTensorCI:
    """A cross interpolation of a coordinate-space function, on a quantics grid.

    Call it with ordinary coordinates -- ``qtt(0.3)``, or ``qtt(0.3, 0.7)`` in two
    dimensions -- and it converts to quantics indices and evaluates the tensor train.
    The underlying objects stay reachable for anything this wrapper does not cover:
    ``.tci`` (the :class:`~qutecipy.tci2.TensorCI2`), ``.grid``, ``.func`` (the
    wrapped quantics-space function, including its cache if caching was on), and
    :meth:`tensortrain`.
    """

    def __init__(self, tci, grid, dtype, ranks=None, errors=None, func=None):
        self.tci = tci
        self.grid = grid
        self.dtype = dtype
        self.ranks = list(ranks) if ranks is not None else []
        self.errors = list(errors) if errors is not None else []
        self.func = func

    # -- structure ---------------------------------------------------------

    def __len__(self) -> int:
        """Number of tensor-train sites (which is not the number of variables)."""
        return len(self.tci)

    def ndims(self) -> int:
        """Number of coordinate variables."""
        return self.grid.ndims()

    def rank(self) -> int:
        return self.tci.rank()

    def linkdims(self) -> list[int]:
        return self.tci.linkdims()

    def tensortrain(self) -> TensorTrain:
        """The interpolation as a plain :class:`~qutecipy.tensortrain.core.TensorTrain`."""
        return TensorTrain(self.tci.sitetensors())

    # -- evaluation --------------------------------------------------------

    def quantics_eval(self, quantics: Sequence[int]):
        """Evaluate at a quantics index string (one index per tensor-train site)."""
        return self.tci.evaluate(list(quantics))

    def grid_eval(self, grididx):
        """Evaluate at an integer grid index (one per variable)."""
        return self.quantics_eval(self.grid.grididx_to_quantics(grididx))

    def eval(self, *coords):
        """Evaluate at original coordinates: ``qtt.eval(x)``, ``qtt.eval(x, y)``, ...

        The coordinate is snapped to the nearest grid point, exactly as
        ``origcoord_to_grididx`` does -- this is an interpolation *on the grid*, so
        off-grid coordinates are answered with their nearest grid point rather than
        by interpolating between points.
        """
        if len(coords) == 1 and isinstance(coords[0], (list, tuple, np.ndarray)):
            coords = tuple(coords[0])
        return self.quantics_eval(self.grid.origcoord_to_quantics(coords))

    def __call__(self, *coords):
        return self.eval(*coords)

    # -- reductions --------------------------------------------------------

    def sum(self):
        """Sum of the interpolation over every point of the grid.

        Linear in the number of sites, not in the number of grid points -- this is
        the tensor-train trick that makes a 2^R-point sum tractable.
        """
        return self.tensortrain().sum()

    def integral(self):
        """Integral over the grid's domain, by the rectangle rule: ``sum() * prod(step)``.

        Accuracy caveat, and it is a real one: this is a *first-order* quadrature, so
        its error is ``O(step) = O(base^-R)`` per dimension and has nothing to do with
        the interpolation ``tolerance``. On the R=20 example in the module docstring
        the interpolation is accurate to 1.1e-9 pointwise while this integral is only
        accurate to 1.3e-6 -- to improve the integral, raise ``R``, not ``tolerance``.
        (For high-accuracy quadrature of a smooth low-dimensional function, use
        :func:`qutecipy.integration.integrate`, which is Gauss-Kronrod based.)

        Only defined for a :class:`~qutecipy.quantics.discretized.DiscretizedGrid`
        with ``includeendpoint=False`` on every axis -- see the raised messages.
        """
        if not isinstance(self.grid, DiscretizedGrid):
            raise TypeError(
                "integral() needs a DiscretizedGrid (a float grid with a step size); "
                f"this is a {type(self.grid).__name__}. Use sum() instead."
            )
        if any(self.grid.includeendpoint):
            # With includeendpoint=True the stored upper_bound is rewritten at
            # construction so the last grid point lands on the user's endpoint. The
            # base^R points then span the *closed* interval, so sum()*step integrates
            # over one step more than the user asked for. Rather than return a
            # quietly-wrong number, say so.
            raise ValueError(
                "integral() implements the half-open rectangle rule, which does not "
                "apply to a grid built with includeendpoint=True (its points span the "
                "closed interval, so sum() * step covers one extra step per such axis). "
                "Rebuild the grid with includeendpoint=False, or use sum() and apply "
                "your own quadrature weights."
            )
        volume = 1.0
        for step in self.grid.grid_step():
            volume *= step
        return self.sum() * volume


def quantics_crossinterpolate(
    dtype,
    f: Callable,
    grid,
    initialpivots: Sequence | None = None,
    cache: bool = True,
    **kwargs,
) -> tuple[QuanticsTensorCI, list[int], list[float]]:
    """Cross interpolate a coordinate-space function on a quantics grid.

    ``f`` takes one argument per grid variable -- ``f(x)`` in one dimension,
    ``f(x, y)`` in two -- in *original coordinates*, not grid or quantics indices.
    ``grid`` is a :class:`~qutecipy.quantics.discretized.DiscretizedGrid` or
    :class:`~qutecipy.quantics.grid.InherentDiscreteGrid`; build it with
    ``.from_resolutions(...)``. ``**kwargs`` go to
    :func:`~qutecipy.tci2.optimize` unchanged (``tolerance``, ``maxbonddim``,
    ``pivotsearch``, ``maxiter``, ...).

    ``initialpivots`` are given in original coordinates too -- a sequence of
    coordinate tuples (or bare scalars in one dimension) -- and are converted to
    quantics indices for you. Choosing one near a feature of ``f`` is the usual way
    to help the pivot search find an isolated peak.

    ``cache`` wraps the quantics-space function in a
    :class:`~qutecipy.tensortrain.cachedfunction.CachedFunction`; it defaults to
    ``True`` here, unlike bare ``crossinterpolate2`` (see the module docstring for
    the measurement behind that choice).

    Returns ``(QuanticsTensorCI, ranks, errors)``.

    >>> import numpy as np
    >>> from qutecipy.quantics import DiscretizedGrid
    >>> grid = DiscretizedGrid.from_resolutions(
    ...     ["x"], [20], lower_bound=(0.0,), upper_bound=(1.0,))
    >>> qtt, ranks, errors = quantics_crossinterpolate(
    ...     np.float64, lambda x: np.exp(-x) * np.cos(20 * x) + 0.3, grid, tolerance=1e-9)
    >>> qtt.rank()  # a 2**20-point function, at bond dimension 3
    3
    """
    localdims = grid.localdimensions()
    qf = quantics_function(dtype, grid, f)
    func = CachedFunction(dtype, qf, localdims) if cache else qf

    pivots = None
    if initialpivots is not None:
        pivots = [
            list(grid.origcoord_to_quantics(
                tuple(p) if isinstance(p, (list, tuple, np.ndarray)) else (p,)
            ))
            for p in initialpivots
        ]

    tci, ranks, errors = crossinterpolate2(dtype, func, localdims, pivots, **kwargs)
    qtt = QuanticsTensorCI(tci, grid, dtype, ranks=ranks, errors=errors, func=func)
    return qtt, ranks, errors
