"""Port of cachedfunction.jl: CachedFunction, memoizing an arbitrary function.

Per CLAUDE.md, the cache is keyed directly by ``tuple(indexset)`` rather than
Julia's mixed-radix integer encoding (`BitIntegers`/`BigInt`) -- simpler, no
overflow-tracking machinery needed (Python's native ``int`` is
arbitrary-precision anyway, so the overflow concern that motivated
`BitIntegers` in Julia doesn't apply here even if we wanted the integer
encoding), and dict-of-tuple lookups are fast enough.

Julia's ``get!(dict, key) do ... end`` compute-if-absent idiom becomes an
explicit ``if key not in cache: cache[key] = ...`` block.
"""
from __future__ import annotations

import itertools
from typing import Callable, Sequence

import numpy as np

from qutecipy.tensortrain.batcheval import BatchEvaluator, isbatchevaluable

# Cache-miss sentinel: distinct from any value a user function could legitimately return
# (including None or NaN), so a single `dict.get` can distinguish "absent" from "cached".
_MISSING = object()


class CachedFunction(BatchEvaluator):
    def __init__(self, dtype, f: Callable, localdims: Sequence[int], cache: dict | None = None):
        self.dtype = dtype
        self.f = f
        self.localdims = list(localdims)
        self.cache: dict[tuple, object] = cache if cache is not None else {}

    def __call__(self, x: Sequence[int]):
        key = tuple(x)
        if len(key) != len(self.localdims):
            raise ValueError("Invalid length of x")
        # One dict lookup on a hit (the common case by construction -- this class only
        # earns its keep when hits dominate) instead of a `in` test followed by a
        # subscript. A miss caches None-valued results correctly too, since absence is
        # signalled by the exception rather than by the value.
        try:
            return self.cache[key]
        except KeyError:
            val = self.f(list(key))
            self.cache[key] = val
            return val

    def __setitem__(self, indexset: Sequence[int], val) -> None:
        self.cache[tuple(indexset)] = val

    def __contains__(self, x: Sequence[int]) -> bool:
        return tuple(x) in self.cache

    def cachedata(self) -> dict[tuple, object]:
        return dict(self.cache)

    def batchevaluate(self, leftindexset: Sequence, rightindexset: Sequence, ncent: int) -> np.ndarray:
        if len(leftindexset) * len(rightindexset) == 0:
            return np.empty((0,) * (ncent + 2), dtype=self.dtype)
        if isbatchevaluable(self.f):
            return self._batcheval_for_batchevaluator(leftindexset, rightindexset, ncent)
        return self._batcheval_default(leftindexset, rightindexset, ncent)

    def _center_combos(self, nl: int, ncent: int):
        center_dims = self.localdims[nl:nl + ncent]
        combos = list(itertools.product(*[range(d) for d in center_dims])) if ncent > 0 else [()]
        return center_dims, combos

    def _batcheval_default(self, leftindexset, rightindexset, ncent) -> np.ndarray:
        nl = len(leftindexset[0])
        center_dims, combos = self._center_combos(nl, ncent)
        lefts = [tuple(l) for l in leftindexset]
        rights = [tuple(r) for r in rightindexset]

        cache = self.cache
        f = self.f
        cacheget = cache.get
        _MISS = _MISSING

        # Flat comprehension instead of per-cell numpy __setitem__ inside a triple loop --
        # cache population is order-independent, so any traversal order is equivalent; this
        # one matches the (i,c,j) axis order used for the final reshape.
        #
        # The hit path is one bound-method dict lookup with a sentinel default; the miss
        # path is pushed into a helper that is only called when the sentinel comes back.
        # (The previous closure cost a Python call plus two dict lookups on *every* cell,
        # which is what made caching a net loss on cheap functions.) The `lk = l + k`
        # prefix is likewise hoisted out of the innermost loop.
        def _miss(key):
            val = f(list(key))
            cache[key] = val
            return val

        flat = [
            val if (val := cacheget(key := lk + r, _MISS)) is not _MISS else _miss(key)
            for lidx in lefts
            for lk in (lidx + k for k in combos)
            for r in rights
        ]
        result = np.array(flat, dtype=self.dtype).reshape(len(lefts), len(combos), len(rights))
        return result.reshape((len(leftindexset), *center_dims, len(rightindexset)))

    def _batcheval_for_batchevaluator(self, leftindexset, rightindexset, ncent) -> np.ndarray:
        nl = len(leftindexset[0])
        center_dims, combos = self._center_combos(nl, ncent)
        result = np.empty((len(leftindexset), len(combos), len(rightindexset)), dtype=self.dtype)
        filled = np.zeros(result.shape, dtype=bool)

        # tuple() conversions and the `left + center` prefix are hoisted out of the
        # innermost loop; the lookup takes a sentinel default so a hit costs one dict
        # probe rather than a `in` test plus a subscript. Loop nesting is reordered to
        # put the varying part innermost -- this pass only fills cells, so order is free.
        lefts = [tuple(l) for l in leftindexset]
        rights = [tuple(r) for r in rightindexset]
        cacheget = self.cache.get
        for i, lidx in enumerate(lefts):
            for c, k in enumerate(combos):
                lk = lidx + k
                for j, r in enumerate(rights):
                    val = cacheget(lk + r, _MISSING)
                    if val is not _MISSING:
                        result[i, c, j] = val
                        filled[i, c, j] = True

        left_needs = [i for i in range(len(leftindexset)) if not filled[i, :, :].all()]
        right_needs = [j for j in range(len(rightindexset)) if not filled[:, :, j].all()]
        if left_needs and right_needs:
            leftindexset_ = [leftindexset[i] for i in left_needs]
            rightindexset_ = [rightindexset[j] for j in right_needs]
            result_ = self.f.batchevaluate(leftindexset_, rightindexset_, ncent)
            cache = self.cache
            for ii, i in enumerate(left_needs):
                lidx = lefts[i]
                for c, k in enumerate(combos):
                    lk = lidx + k
                    sub = result_[(ii, *k)]
                    for jj, j in enumerate(right_needs):
                        val = sub[jj]
                        cache[lk + rights[j]] = val
                        if not filled[i, c, j]:
                            result[i, c, j] = val
                            filled[i, c, j] = True

        unfilled = np.argwhere(~filled)
        if unfilled.size:
            cache = self.cache
            for i, c, j in unfilled:
                result[i, c, j] = cache[lefts[i] + combos[c] + rights[j]]

        return result.reshape((len(leftindexset), *center_dims, len(rightindexset)))
