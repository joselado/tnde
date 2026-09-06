"""Port of globalpivotfinder.jl + globalsearch.jl.

Two genuinely different greedy search variants (per CLAUDE.md, kept separate
rather than merged): ``DefaultGlobalPivotFinder`` (random restarts, single
one-pass greedy coordinate scan, keep-best-if-above-margin) vs.
``_floatingzone``/``estimate_true_error`` (unconditional coordinate ascent to
a true local error maximum, used for diagnostics and by the older
``search_global_pivots``).
"""
from __future__ import annotations

import random as _random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from qutecipy.tensortrain.cache import TTCache
from qutecipy.tensortrain.core import TensorTrain


@dataclass
class GlobalPivotSearchInput:
    localdims: list[int]
    current_tt: TensorTrain
    maxsamplevalue: float
    Iset: list[list[tuple]]
    Jset: list[list[tuple]]


class AbstractGlobalPivotFinder(ABC):
    @abstractmethod
    def __call__(
        self, input: GlobalPivotSearchInput, f: Callable, abstol: float,
        verbosity: int = 0, rng: _random.Random | None = None,
    ) -> list[tuple]: ...


class _OneSiteScan:
    """All TT values on the "cross" through a base point: for a fixed point and site
    ``p``, every value obtained by varying only the p-th index.

    The greedy coordinate scan below asks for exactly that, one site at a time, so
    evaluating the full chain per candidate value re-multiplies L-1 identical factors
    every time. Here the untouched factors are contracted once into a left and a right
    environment, and the whole row of predictions then costs two matrix products
    regardless of the local dimension.

    Values are mathematically identical to ``tt(point)``, but the environments are
    accumulated outward from the ends rather than strictly left to right, so they can
    differ in the last ulp -- fine here, where they feed a >-comparison of interpolation
    errors in a search that is randomized to begin with. Falls back to plain per-point
    evaluation for anything that isn't a 3-leg (single physical index per site) TT.
    """

    def __init__(self, tt):
        self.tt = tt
        sts = tt.sitetensors()
        self.sts = sts
        # The environment recursion seeds both ends with a length-1 vector, so it needs
        # the usual trivial boundary bonds on top of one physical index per site.
        self.simple = (
            all(T.ndim == 3 for T in sts) and sts[0].shape[0] == 1 and sts[-1].shape[-1] == 1
        )
        self._point: list | None = None

    def _build_environments(self, point) -> None:
        sts = self.sts
        n = len(sts)
        # left[p] = product of sites 0..p-1 at `point`; right[p] = product of sites p+1..n-1.
        left: list = [None] * n
        acc = np.ones(1, dtype=sts[0].dtype)
        for p in range(n):
            left[p] = acc
            acc = acc @ sts[p][:, point[p], :]
        right: list = [None] * n
        acc = np.ones(1, dtype=sts[-1].dtype)
        for p in range(n - 1, -1, -1):
            right[p] = acc
            acc = sts[p][:, point[p], :] @ acc
        self._left, self._right = left, right
        self._point = list(point)

    def __call__(self, point, p: int) -> np.ndarray:
        if not self.simple:
            return np.array([
                self.tt(list(point[:p]) + [v] + list(point[p + 1:]))
                for v in range(self.sts[p].shape[1])
            ])
        if self._point != list(point):
            self._build_environments(point)
        T = self.sts[p]
        # (l, d, r) x (r,) -> (l, d), then (l,) x (l, d) -> (d,)
        return self._left[p] @ (T.reshape(-1, T.shape[2]) @ self._right[p]).reshape(T.shape[0], T.shape[1])


class DefaultGlobalPivotFinder(AbstractGlobalPivotFinder):
    def __init__(self, nsearch: int = 5, maxnglobalpivot: int = 5, tolmarginglobalsearch: float = 10.0,
                 npivotseed: int = 2):
        self.nsearch = nsearch
        self.maxnglobalpivot = maxnglobalpivot
        self.tolmarginglobalsearch = tolmarginglobalsearch
        self.npivotseed = npivotseed

    def _pivot_seeds(self, input: GlobalPivotSearchInput) -> list[list[int]]:
        """Search starting points derived from the pivots already found, plus their
        index reflections.

        Uniformly random starting points are a poor way to look for a region the
        interpolation gets wrong, because they say nothing about where the function
        actually lives. For a function supported on a small fraction of the index space
        -- a Gaussian on a 2**30 quantics grid, say -- a random point lands in the
        numerically-zero region essentially always, and since the scan below only visits
        points differing from its start in *one* coordinate, it never leaves that region.

        The pivots are the only known-good locations, and this class is already handed
        them. Their *reflections* (digit ``d`` -> ``localdim-1-d``) matter as much: when a
        function's support straddles a high-order digit boundary, the two halves differ in
        every digit, so no single-coordinate move connects them. Concretely, a state
        centred on a 2**30 grid has its pivot at ``100...0`` and the branch that gets
        missed at ``011...1``. Without this, that branch is silently interpolated as zero
        while TCI reports convergence -- measured, 3 runs in 6.
        """
        L = len(input.localdims)
        seeds: list[list[int]] = []
        for i in range(len(input.Iset)):
            if len(seeds) >= 2 * self.npivotseed:
                break
            if not input.Iset[i] or not input.Jset[i]:
                continue
            head, tail = tuple(input.Iset[i][0]), tuple(input.Jset[i][0])
            if len(head) + 1 + len(tail) != L:
                continue
            point = list(head + (0,) + tail)
            seeds.append(point)
            seeds.append([input.localdims[k] - 1 - v for k, v in enumerate(point)])
        return seeds

    def __call__(
        self, input: GlobalPivotSearchInput, f: Callable, abstol: float,
        verbosity: int = 0, rng: _random.Random | None = None,
    ) -> list[tuple]:
        rng = rng if rng is not None else _random
        L = len(input.localdims)

        initial_points = [[rng.randrange(input.localdims[p]) for p in range(L)] for _ in range(self.nsearch)]
        initial_points += self._pivot_seeds(input)

        predict = _OneSiteScan(input.current_tt)

        found_pivots: list[tuple] = []
        for point in initial_points:
            current_point = list(point)
            best_error = 0.0
            best_point = list(point)

            for p in range(L):
                # Every point visited in the inner loop differs from `point` in at most the
                # p-th coordinate, so all site tensors but the p-th contribute the same
                # factors throughout it. _OneSiteScan contracts those into a left/right
                # environment pair once per (point, p) and returns all localdims[p]
                # predictions at once, instead of walking the whole chain per candidate.
                predictions = predict(point, p)
                for v in range(input.localdims[p]):
                    current_point[p] = v
                    error = abs(f(current_point) - predictions[v])
                    if error > best_error:
                        best_error = error
                        best_point = list(current_point)
                current_point[p] = point[p]

            if best_error > abstol * self.tolmarginglobalsearch:
                found_pivots.append(tuple(best_point))

        if len(found_pivots) > self.maxnglobalpivot:
            found_pivots = found_pivots[: self.maxnglobalpivot]

        if verbosity > 0:
            print(f"Found {len(found_pivots)} global pivots")

        return found_pivots


def _floatingzone(
    ttcache: TTCache, f: Callable, earlystoptol: float = float("inf"), nsweeps: int = 10**9,
    initp: Sequence[int] | None = None,
) -> tuple[tuple, float]:
    if nsweeps <= 0:
        raise ValueError("nsweeps should be positive!")

    localdims = [d[0] for d in ttcache.sitedims()]
    n = len(ttcache)

    pivot = list(initp) if initp is not None else [_random.randrange(d) for d in localdims]

    dtype = ttcache.dtype
    maxerror = abs(f(pivot) - ttcache(pivot))

    for _ in range(nsweeps):
        prev_maxerror = maxerror
        for ipos in range(n):
            from qutecipy.tci2 import filltensor  # local import: avoids a tci2<->globalpivot import cycle

            left = [tuple(pivot[:ipos])]
            right = [tuple(pivot[ipos + 1:])]
            exactdata = np.asarray(filltensor(dtype, f, localdims, left, right, 1))
            prediction = np.asarray(filltensor(dtype, ttcache, localdims, left, right, 1))
            err = np.abs(exactdata - prediction).reshape(-1)
            pivot[ipos] = int(np.argmax(err))
            maxerror = max(float(np.max(err)), maxerror)

        if maxerror == prev_maxerror or maxerror > earlystoptol:
            break

    return tuple(pivot), maxerror


def estimate_true_error(
    tt: TensorTrain, f: Callable, nsearch: int = 100, initialpoints: Sequence[Sequence[int]] | None = None,
) -> list[tuple[tuple, float]]:
    if nsearch <= 0 and initialpoints is None:
        raise ValueError("No search is performed")
    if nsearch < 0:
        raise ValueError("nsearch must be non-negative")

    if nsearch > 0 and initialpoints is None:
        initialpoints = [[_random.randrange(d[0]) for d in tt.sitedims()] for _ in range(nsearch)]

    ttcache = TTCache.from_tt(tt)
    pivoterror = [_floatingzone(ttcache, f, initp=initp) for initp in initialpoints]
    pivoterror.sort(key=lambda pe: pe[1], reverse=True)

    seen = set()
    result = []
    for p, e in pivoterror:
        if p not in seen:
            seen.add(p)
            result.append((p, e))
    return result
