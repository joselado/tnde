"""Quantics grids in any number of dimensions.

A ``d``-dimensional box discretised with ``2**R`` points per direction is stored as an
*interleaved* quantics tensor train over ``d * R`` binary sites,

    x_1 y_1 z_1  x_2 y_2 z_2  ...  x_R y_R z_R

with ``x_1`` the most significant bit of the ``x`` index. Interleaving puts the digits
of all variables that share a length scale on neighbouring sites, which is what keeps
the bond dimension of a smooth ``d``-dimensional function small.

The grid is *periodic* and excludes the upper endpoint: ``x_j = xmin + j L / 2**R``,
so the spacing is ``L / 2**R`` and the discrete Fourier transform of the train has
period exactly ``L``. That makes the momentum of DFT index ``j`` the usual
``k_j = 2 pi j / L`` for ``j < 2**(R-1)`` and ``2 pi (j - 2**R) / L`` above -- the map
:meth:`Grid.momentum` implements. (The legacy Gross-Pitaevskii drivers use a grid that
*includes* the endpoint; do not mix the two.)

Indices are always flat 64-bit integers into the interleaved train, so a grid can have
at most 62 sites: ``R = 31`` in 1D, ``R = 20`` in 3D.
"""
from __future__ import annotations

from itertools import product

import numpy as np


class Grid:
    """A periodic ``2**R``-per-direction quantics grid on a box.

    ``bounds`` is ``(lo, hi)`` for one dimension or a sequence of such pairs.
    """

    def __init__(self, R: int, bounds):
        bounds = list(bounds)
        if len(bounds) == 2 and np.isscalar(bounds[0]):
            bounds = [tuple(bounds)]
        self.R = int(R)
        self.bounds = [(float(lo), float(hi)) for lo, hi in bounds]
        self.ndim = len(self.bounds)
        self.nsites = self.ndim * self.R
        if self.nsites > 62:
            raise ValueError(f"{self.ndim} x R={self.R} = {self.nsites} sites exceeds "
                             "the 62-bit index limit")
        if any(hi <= lo for lo, hi in self.bounds):
            raise ValueError("every bound must satisfy lo < hi")
        self.M = 1 << self.R
        self.lengths = [hi - lo for lo, hi in self.bounds]
        self.spacings = [L / self.M for L in self.lengths]
        self.dvol = float(np.prod(self.spacings))
        self.localdims = [2] * self.nsites

    def __repr__(self):
        b = ", ".join(f"[{lo:g}, {hi:g})" for lo, hi in self.bounds)
        return f"Grid(R={self.R}, {self.ndim}D, {b}, {self.M}^{self.ndim} points)"

    # ------------------------------------------------------------------
    # flat interleaved index <-> per-variable indices
    # ------------------------------------------------------------------

    def fuse(self, *ij) -> np.ndarray:
        """The flat interleaved index of per-variable grid indices ``ij[v]``."""
        if len(ij) != self.ndim:
            raise ValueError(f"expected {self.ndim} index arrays, got {len(ij)}")
        ij = [np.asarray(a, dtype=np.int64) for a in ij]
        if self.ndim == 1:
            return ij[0].copy()
        out = np.zeros(np.broadcast(*ij).shape, dtype=np.int64)
        d, R = self.ndim, self.R
        for v, a in enumerate(ij):
            for n in range(R):
                out |= ((a >> (R - 1 - n)) & 1) << (d * R - 1 - (d * n + v))
        return out

    def split(self, idx) -> tuple[np.ndarray, ...]:
        """Per-variable grid indices of flat interleaved indices."""
        idx = np.asarray(idx, dtype=np.int64)
        if self.ndim == 1:
            return (idx,)
        d, R = self.ndim, self.R
        out = []
        for v in range(d):
            a = np.zeros_like(idx)
            for n in range(R):
                a |= ((idx >> (d * R - 1 - (d * n + v))) & 1) << (R - 1 - n)
            out.append(a)
        return tuple(out)

    def bits(self, idx) -> list[list[int]]:
        """Flat indices as lists of site digits -- the form TCI takes pivots in."""
        idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
        n = self.nsites
        return [[(int(i) >> (n - 1 - s)) & 1 for s in range(n)] for i in idx]

    # ------------------------------------------------------------------
    # coordinates
    # ------------------------------------------------------------------

    def coords(self, idx) -> tuple[np.ndarray, ...]:
        """Real-space coordinates ``(x, y, ...)`` of flat indices."""
        js = self.split(idx)
        return tuple(lo + j.astype(np.float64) * h
                     for j, (lo, _), h in zip(js, self.bounds, self.spacings))

    def coord_of(self, i: int) -> tuple[float, ...]:
        """Coordinates of one flat index, in pure Python -- for the single-point
        evaluations TCI's pivot searches make, where array overhead dominates."""
        i = int(i)
        d, R = self.ndim, self.R
        if d == 1:
            return (self.bounds[0][0] + i * self.spacings[0],)
        out = []
        for v in range(d):
            j = 0
            for n in range(R):
                j |= ((i >> (d * R - 1 - (d * n + v))) & 1) << (R - 1 - n)
            out.append(self.bounds[v][0] + j * self.spacings[v])
        return tuple(out)

    def momentum(self, j, v: int) -> np.ndarray:
        """Physical momentum of DFT index ``j`` along variable ``v``."""
        j = np.asarray(j, dtype=np.float64)
        return 2.0 * np.pi / self.lengths[v] * np.where(j < self.M // 2, j, j - self.M)

    def momenta(self, idx, reversed_slots: bool = False) -> tuple[np.ndarray, ...]:
        """Physical momenta ``(k_x, k_y, ...)`` of flat indices into a momentum-space
        train.

        ``reversed_slots=True`` reads the train in the order :func:`tnde.pde.fourier_transform`
        produces, where slot ``s`` holds variable ``ndim - 1 - s``.
        """
        js = self.split(idx)
        d = self.ndim
        return tuple(self.momentum(js[d - 1 - v] if reversed_slots else js[v], v)
                     for v in range(d))

    def nearest(self, *x) -> np.ndarray:
        """Flat index of the grid point nearest to real coordinates ``x[v]``."""
        if len(x) != self.ndim:
            raise ValueError(f"expected {self.ndim} coordinates, got {len(x)}")
        js = []
        for a, (lo, _), h in zip(x, self.bounds, self.spacings):
            j = np.rint((np.asarray(a, dtype=np.float64) - lo) / h).astype(np.int64)
            js.append(np.clip(j, 0, self.M - 1))
        return self.fuse(*js)

    # ------------------------------------------------------------------
    # dense and sampled views
    # ------------------------------------------------------------------

    def mesh(self) -> tuple[np.ndarray, ...]:
        """Full coordinate arrays of shape ``(M,) * ndim`` (``indexing='ij'``).
        Only for small grids."""
        axes = [lo + np.arange(self.M) * h for (lo, _), h in zip(self.bounds, self.spacings)]
        return tuple(np.meshgrid(*axes, indexing="ij"))

    def kmesh(self) -> tuple[np.ndarray, ...]:
        """Full momentum arrays in DFT (``numpy.fft``) ordering, shape ``(M,) * ndim``."""
        axes = [self.momentum(np.arange(self.M), v) for v in range(self.ndim)]
        return tuple(np.meshgrid(*axes, indexing="ij"))

    def dense_indices(self) -> np.ndarray:
        """Flat indices of every grid point, shaped ``(M,) * ndim``."""
        ax = np.arange(self.M)
        return self.fuse(*np.meshgrid(*([ax] * self.ndim), indexing="ij"))

    def sample(self, n: int = 256, window=None):
        """``n`` points per direction, evenly spread over the box (or over ``window``,
        a sequence of ``(lo, hi)`` pairs). Returns ``(idx, coords)`` with ``idx`` of
        shape ``(n,) * ndim`` and ``coords`` a tuple of 1D axis arrays."""
        if window is None:
            window = self.bounds
        elif self.ndim == 1 and np.isscalar(window[0]):
            window = [tuple(window)]
        js, axes = [], []
        for v, ((wlo, whi), (lo, _), h) in enumerate(zip(window, self.bounds, self.spacings)):
            j = np.rint((np.linspace(wlo, whi, n, endpoint=False) - lo) / h).astype(np.int64)
            j = np.clip(j, 0, self.M - 1)
            js.append(j)
            axes.append(lo + j * h)
        idx = self.fuse(*np.meshgrid(*js, indexing="ij"))
        return idx, tuple(axes)

    # ------------------------------------------------------------------
    # pivots
    # ------------------------------------------------------------------

    def centre_pivots(self) -> list[list[int]]:
        """The ``2**ndim`` grid points straddling the box centre, as TCI pivots.

        A function centred in the box sits exactly on the most significant bit of every
        variable; seeding both sides of each such boundary keeps TCI from interpolating
        only one branch (see :func:`tnde.operators.peak_pivots`).
        """
        c = self.M // 2
        pts = [self.fuse(*combo) for combo in product((c - 1, c), repeat=self.ndim)]
        return self.bits(np.array(pts))

    def corner_pivots(self) -> list[list[int]]:
        """The ``2**ndim`` corners of the index box -- where small momenta live on a
        momentum-space train."""
        pts = [self.fuse(*combo) for combo in product((0, self.M - 1), repeat=self.ndim)]
        return self.bits(np.array(pts))
