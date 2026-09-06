"""Array-valued (vector-/matrix-valued) tensor cross interpolation.

**This module is an extension: it has no counterpart in
TensorCrossInterpolation.jl.** Everything else in ``qutecipy`` is a faithful
port of a specific Julia file; this one is additive, and deliberately does not
modify any ported module -- it drives the unmodified ``crossinterpolate2`` on a
slightly larger index space.

Problem
-------
Interpolate ``f(x) -> np.ndarray`` of a fixed shape ``S`` (e.g. ``(K,)`` or
``(m, n)``), where all components share roughly the same structure, so a single
set of pivots should serve all of them.

Approach: the component index as an extra site
----------------------------------------------
We interpolate the *scalar* joint function ::

    F(x_0, ..., x_{N-1}, k) = f(x).reshape(-1)[k]

on the extended chain ``localdims + [K]`` (with ``K = prod(S)``), using the
existing scalar TCI2 verbatim. The component index becomes an ordinary
physical leg on an extra edge site. This is the standard tensor-train practice
of carrying a discrete component/orbital index as an additional leg.

Because the extra site sits at one *end* of the chain, all interior index sets
``Iset[b]``/``Jset[b]`` -- i.e. the pivots -- are shared across components by
construction, and one tensor train per component is recovered exactly by
slicing the edge core at ``k`` and absorbing it into its neighbour
(:meth:`ArrayTensorTrain.component`). Site tensors ``0 .. N-2`` are then
*identical* across all components; only the last one differs. So this delivers
the "one MPS per component, same pivots for all" shape, while the pivot search
itself sees every component.

Why not pivot on a scalar surrogate
-----------------------------------
The obvious shortcut -- run ordinary TCI2 on ``g(x) = max_k |f(x)_k|`` (or a
random projection), then rebuild one TT per component from the resulting
``Iset``/``Jset`` -- is not used here, because nothing then guarantees that each
component's own pivot matrix ``P_k = f_k(Iset[b+1], Jset[b])`` is well
conditioned. ``TensorCI2.compute_sitetensor`` inverts that matrix
(``np.linalg.solve``), so a component that happens to be flat or near-zero on
the shared pivots is destroyed *silently* -- the reported error only ever
measured the surrogate. Block/vector ACA (shared pivot chosen from an
aggregated residual, per-component rank-1 deflation) has the same weakness: the
deflation divides by that component's own pivot value.

With the extra-site formulation every matrix that gets inverted is a submatrix
of the *joint* tensor, selected by rrLU's own pivoting, so it is well
conditioned by construction, and the error TCI2 controls (bond errors, global
pivot search) is the true joint residual *including* the component axis.

Cost
----
At any cut, ``max_k chi_k <= chi_joint <= sum_k chi_k``. The user assumption
"the components share structure" is exactly the statement that
``chi_joint ~ max_k chi_k``, which is where this wins over ``K`` independent
TCI runs: the number of distinct ``x`` visited stays comparable to a single
run, and :class:`ArrayValuedFunction` caches by ``x``, so all ``K`` components
come from *one* call to the user's ``f``. If the components do not share
structure, this degrades gracefully (fatter bonds, still accurate) rather than
silently losing accuracy.

Error normalization, and why weights are on by default
------------------------------------------------------
With ``normalizeerror=True`` (TCI2's default), ``tolerance`` is relative to the
single largest sampled value across *all* components. Unweighted, a component
orders of magnitude smaller than the largest is therefore simply dropped -- and
dropped *silently*, since the reported error is dominated by the big component.
For ``f = [exp(-2x), 1e-8*(cos(5x)+1.5)]`` at ``tolerance=1e-6`` the joint TT
collapses to bond dimension 1 and the second component comes out ~80% wrong.

So ``crossinterpolate2_array`` defaults to ``componentweights="auto"``: it
samples ``f`` at a few dozen points up front, takes ``w_k = max_x |f_k(x)|``,
interpolates ``f_k / w_k``, and multiplies the weights back on the way out.
``tolerance`` then means the same thing for every component. The sampling is
cached in the same adapter TCI2 uses, so those calls are not extra work.

Pass ``componentweights=None`` for the old unweighted behaviour, or an explicit
array when the scales are known analytically. Weighting is not free: forcing a
genuinely negligible component to full *relative* accuracy costs bond dimension,
so ``None`` is the right choice when the small components genuinely do not
matter.
"""
from __future__ import annotations

import itertools
import random as _random
from typing import Callable, Sequence

import numpy as np

from qutecipy.tci2 import crossinterpolate2
from qutecipy.tensortrain.base import AbstractTensorTrain
from qutecipy.tensortrain.batcheval import BatchEvaluator
from qutecipy.tensortrain.core import TensorTrain

_POSITIONS = ("last", "first")


def _normalize_valueshape(valueshape) -> tuple[int, ...]:
    if isinstance(valueshape, (int, np.integer)):
        valueshape = (int(valueshape),)
    shape = tuple(int(s) for s in valueshape)
    if len(shape) == 0:
        raise ValueError("valueshape must have at least one axis; use crossinterpolate2 for scalar f.")
    if any(s <= 0 for s in shape):
        raise ValueError(f"valueshape entries must be positive, got {shape}.")
    return shape


def _check_position(componentposition: str) -> str:
    if componentposition not in _POSITIONS:
        raise ValueError(f"Unknown componentposition {componentposition!r}. Choose from {_POSITIONS}.")
    return componentposition


class ArrayValuedFunction(BatchEvaluator):
    """Adapter turning an array-valued ``f(x) -> ndarray of shape S`` into the
    scalar function of an extended index set that scalar TCI2 consumes.

    ``self(indexset)`` takes ``len(localdims) + 1`` indices: the ``x`` indices
    plus one component index ``k`` (flattened over ``S``), appended at the end
    (``componentposition="last"``, the default) or prepended
    (``"first"``).

    The adapter caches the whole array per ``x``, so the ``K`` scalar queries
    that TCI2 makes at a given ``x`` cost a single call to the user's ``f``.
    (``CachedFunction`` is *not* a substitute here: it keys on the full index
    including ``k``, so it would still call ``f`` once per component.)
    """

    def __init__(
        self,
        dtype,
        f: Callable,
        localdims: Sequence[int],
        valueshape,
        componentposition: str = "last",
        componentweights=None,
    ):
        self.dtype = np.dtype(dtype)
        self.f = f
        self.localdims = [int(d) for d in localdims]
        if len(self.localdims) == 0:
            raise ValueError("localdims must not be empty.")
        self.valueshape = _normalize_valueshape(valueshape)
        self.K = int(np.prod(self.valueshape))
        self.componentposition = _check_position(componentposition)
        self.nsites = len(self.localdims)

        if componentposition == "last":
            self.extendedlocaldims = self.localdims + [self.K]
        else:
            self.extendedlocaldims = [self.K] + self.localdims

        # cache[x] = f(x), flattened, *unweighted* (weights are applied per call,
        # a single scalar multiply, so that the cache holds the raw user values --
        # which is also what lets set_componentweights work after sampling).
        self.cache: dict[tuple, np.ndarray] = {}
        self.ncalls = 0     # calls to the user's f (== number of distinct x seen)
        self.nqueries = 0   # scalar (x, k) queries made by TCI2

        self.set_componentweights(componentweights)

    def set_componentweights(self, componentweights) -> None:
        """Install (or clear, with ``None``) the per-component weights.

        Safe to call after values have been cached: the cache holds *unweighted*
        user values and the weights are applied per query. This is what lets
        :func:`estimate_componentweights` sample through this same adapter -- so
        the sampling calls are not wasted, they are exactly the cache entries TCI2
        would otherwise pay for later.
        """
        if componentweights is None:
            self.componentweights = None
            self._invweights = None
            return
        w = np.asarray(componentweights).reshape(-1)
        if w.size != self.K:
            raise ValueError(f"componentweights must have {self.K} entries, got {w.size}.")
        if np.any(w == 0):
            raise ValueError("componentweights must all be nonzero.")
        self.componentweights = w
        self._invweights = 1.0 / w

    def values(self, x: Sequence[int]) -> np.ndarray:
        """All components at ``x``, flattened. One call to ``f`` per distinct ``x``."""
        key = tuple(x)
        cached = self.cache.get(key)
        if cached is None:
            val = np.asarray(self.f(list(key)), dtype=self.dtype)
            if val.shape != self.valueshape:
                raise ValueError(f"f returned shape {val.shape}, expected {self.valueshape}.")
            cached = val.reshape(-1)
            self.cache[key] = cached
            self.ncalls += 1
        return cached

    def split(self, indexset: Sequence[int]) -> tuple[tuple[int, ...], int]:
        """Split an extended index into its ``x`` part and its component index."""
        if len(indexset) != self.nsites + 1:
            raise ValueError(f"Expected {self.nsites + 1} indices, got {len(indexset)}.")
        if self.componentposition == "last":
            return tuple(indexset[: self.nsites]), int(indexset[self.nsites])
        return tuple(indexset[1:]), int(indexset[0])

    def __call__(self, indexset: Sequence[int]):
        self.nqueries += 1
        x, k = self.split(indexset)
        v = self.values(x)[k]
        return v if self._invweights is None else v * self._invweights[k]

    # -- batch evaluation --------------------------------------------------

    def _component_slot(self, nl: int, ncent: int) -> tuple[str, int]:
        """Which of the (left, center, right) blocks carries the component index,
        and at which offset inside that block."""
        pos = 0 if self.componentposition == "first" else self.nsites
        if pos < nl:
            return "left", pos
        if pos < nl + ncent:
            return "center", pos - nl
        return "right", pos - nl - ncent

    def batchevaluate(self, leftindexset: Sequence, rightindexset: Sequence, ncent: int) -> np.ndarray:
        """Evaluate a whole (left x center x right) batch of *extended* indices at once.

        This is what makes the extra-site formulation cheap. TCI2 asks for the same
        ``x`` once per component (the component leg is an ordinary site, so a batch
        that spans it contains ``K`` cells per distinct ``x``), and going through
        :meth:`__call__` costs three Python calls per cell. Here the component slot
        is stripped out of whichever block holds it, ``values`` is called once per
        distinct ``x`` -- a factor ``K`` fewer Python-level steps -- and the batch is
        assembled with a single numpy gather.

        Values are identical to the generic per-cell path, cell for cell; only the
        bookkeeping around ``f`` changes. ``f`` itself is called exactly as often
        either way (``values`` memoizes per ``x``).
        """
        nA, nC = len(leftindexset), len(rightindexset)
        if nA * nC == 0:
            return np.empty((0,) * (ncent + 2), dtype=self.dtype)

        nl = len(leftindexset[0])
        centerdims = self.extendedlocaldims[nl:nl + ncent]
        centers = list(itertools.product(*[range(d) for d in centerdims])) if ncent > 0 else [()]
        blocks = [[tuple(t) for t in leftindexset], centers, [tuple(t) for t in rightindexset]]

        which, offset = self._component_slot(nl, ncent)
        b = {"left": 0, "center": 1, "right": 2}[which]

        # Strip the component coordinate out of its block: `kvals` is that block's
        # component index per element, `reduced` the same block with the slot removed.
        # Distinct reduced entries are what actually index the user's f.
        kvals = np.fromiter((t[offset] for t in blocks[b]), dtype=np.intp, count=len(blocks[b]))
        uniq: dict[tuple, int] = {}
        rid = np.empty(len(blocks[b]), dtype=np.intp)
        for pos_, t in enumerate(blocks[b]):
            red = t[:offset] + t[offset + 1:]
            got = uniq.get(red)
            if got is None:
                got = uniq[red] = len(uniq)
            rid[pos_] = got
        blocks[b] = list(uniq)

        nAp, nBp, nCp = (len(x) for x in blocks)
        # One row of V per distinct x in this batch, laid out in C order over the
        # three (post-reduction) blocks so the row index is pure arithmetic below.
        V = np.empty((nAp * nBp * nCp, self.K), dtype=self.dtype)
        row = 0
        for ta in blocks[0]:
            for tb in blocks[1]:
                tab = ta + tb
                for tc in blocks[2]:
                    V[row] = self.values(tab + tc)
                    row += 1

        eff = [np.arange(len(blk), dtype=np.intp) for blk in (leftindexset, centers, rightindexset)]
        eff[b] = rid
        rowidx = (eff[0][:, None, None] * nBp + eff[1][None, :, None]) * nCp + eff[2][None, None, :]
        kidx = kvals.reshape([(-1 if i == b else 1) for i in range(3)])

        result = V[rowidx, kidx]
        if self._invweights is not None:
            result = result * self._invweights[kidx]
        self.nqueries += result.size
        return result.astype(self.dtype, copy=False).reshape((nA, *centerdims, nC))

    def extend_pivot(self, x: Sequence[int]) -> list[tuple[int, ...]]:
        """One extended pivot per component, all sitting at the same ``x``."""
        x = tuple(int(i) for i in x)
        if len(x) != self.nsites:
            raise ValueError(f"Expected {self.nsites} indices, got {len(x)}.")
        if self.componentposition == "last":
            return [x + (k,) for k in range(self.K)]
        return [(k,) + x for k in range(self.K)]

    def clear(self) -> None:
        """Drop the cached function values (they grow as ndistinct_x * K)."""
        self.cache.clear()


def estimate_componentweights(
    fa: "ArrayValuedFunction",
    nsample: int = 50,
    xsamples: Sequence[Sequence[int]] | None = None,
    floor: float = 1e-14,
    rng=None,
) -> np.ndarray:
    """Estimate a per-component scale for ``fa``, for use as ``componentweights``.

    Samples ``f`` at ``xsamples`` plus randomly drawn ``x`` until ``nsample``
    distinct points have been seen, and returns ``w_k = max_x |f_k(x)|`` over that
    sample. Interpolating ``f_k / w_k`` then puts every component on a scale of
    order 1, which is what makes a relative ``tolerance`` mean the same thing for
    all of them (see the "Error normalization caveat" in the module docstring).

    The sampling goes through ``fa.values``, so every call is cached: these are not
    throwaway evaluations, they are cache entries TCI2 would otherwise pay for.

    Two guards on the returned weights, both about not amplifying a component whose
    sampled scale is misleadingly small:

    - a component that is *exactly* zero on every sampled point gets the global max
      instead of a near-zero weight, i.e. it keeps the unweighted behaviour rather
      than having whatever appears later blown up;
    - any other component is floored at ``floor * max_k w_k``, bounding the
      amplification at ``1 / floor``.

    This is a heuristic over a finite sample: if it misses the region where a
    component is largest, that component ends up over-weighted (harmless -- it just
    gets sampled at values above 1). Pass explicit ``componentweights`` when the
    scales are known analytically.
    """
    rng = rng if rng is not None else _random
    xs: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for x in xsamples or []:
        key = tuple(int(i) for i in x)
        if len(key) != fa.nsites:
            raise ValueError(f"Sample point {key} has length {len(key)}, expected {fa.nsites}.")
        if key not in seen:
            seen.add(key)
            xs.append(key)

    total = 1
    for d in fa.localdims:
        total *= d
    want = min(max(int(nsample), 0), total)
    # Bounded rejection sampling: `want <= total` guarantees the target is reachable,
    # the guard just keeps a pathologically unlucky draw sequence from spinning.
    attempts = 0
    while len(xs) < want and attempts < 20 * want + 100:
        attempts += 1
        key = tuple(rng.randrange(d) for d in fa.localdims)
        if key not in seen:
            seen.add(key)
            xs.append(key)

    if not xs:
        raise ValueError("No sample points to estimate componentweights from.")

    scale = np.zeros(fa.K, dtype=float)
    for key in xs:
        np.maximum(scale, np.abs(fa.values(key)), out=scale)

    smax = float(scale.max())
    if smax == 0.0:
        raise ValueError(
            "Every component is zero at all sampled points; cannot derive componentweights."
        )
    return np.where(scale > 0.0, np.maximum(scale, floor * smax), smax)


class ArrayTensorTrain:
    """A tensor train with one free output leg: evaluates to an array of shape ``S``.

    Wraps the ``N+1``-site tensor train produced on the extended chain (``N``
    ``x`` sites plus the component site at one end). ``att(x)`` returns the
    interpolated array; ``att.component(k)`` returns an ordinary ``N``-site
    :class:`~qutecipy.tensortrain.core.TensorTrain` for one component.
    """

    def __init__(
        self,
        tt: AbstractTensorTrain,
        valueshape,
        componentposition: str = "last",
        componentweights=None,
    ):
        self.tt = tt
        self.valueshape = _normalize_valueshape(valueshape)
        self.K = int(np.prod(self.valueshape))
        self.componentposition = _check_position(componentposition)
        self.componentweights = None if componentweights is None else np.asarray(componentweights).reshape(-1)
        if self.componentweights is not None and self.componentweights.size != self.K:
            raise ValueError(f"componentweights must have {self.K} entries.")

        cores = tt.sitetensors()
        self.nsites = len(cores) - 1
        if self.nsites < 1:
            raise ValueError("An array-valued tensor train needs at least one x site plus the component site.")
        compcore = cores[-1] if componentposition == "last" else cores[0]
        if compcore.shape[1] != self.K:
            raise ValueError(
                f"Component site has physical dimension {compcore.shape[1]}, expected {self.K}."
            )

    # -- accessors ---------------------------------------------------------

    def sitetensors(self) -> list[np.ndarray]:
        """The underlying extended (``N+1``-site) site tensors."""
        return self.tt.sitetensors()

    def localdims(self) -> list[int]:
        cores = self.sitetensors()
        xcores = cores[:-1] if self.componentposition == "last" else cores[1:]
        return [T.shape[1] for T in xcores]

    def linkdims(self) -> list[int]:
        return self.tt.linkdims()

    def rank(self) -> int:
        return self.tt.rank()

    def __len__(self) -> int:
        return self.nsites

    def _flatindex(self, k) -> int:
        if isinstance(k, (int, np.integer)):
            idx = int(k)
            if not 0 <= idx < self.K:
                raise IndexError(f"Component index {idx} out of range for {self.K} components.")
            return idx
        return int(np.ravel_multi_index(tuple(int(i) for i in k), self.valueshape))

    # -- evaluation --------------------------------------------------------

    def _contract(self, mats: Sequence[np.ndarray]) -> np.ndarray:
        """Contract per-x-site matrices with the component core, leaving the
        component leg free. Returns a flat length-K vector."""
        cores = self.sitetensors()
        if self.componentposition == "last":
            v = np.eye(1, dtype=cores[0].dtype)
            for mat in mats:
                v = v @ mat
            return (v @ cores[-1][:, :, 0]).reshape(-1)
        v = None
        for mat in mats:
            v = mat if v is None else v @ mat
        return (cores[0][0, :, :] @ v).reshape(-1)

    def _rescale(self, flat: np.ndarray) -> np.ndarray:
        if self.componentweights is not None:
            flat = flat * self.componentweights
        return flat.reshape(self.valueshape)

    def evaluate(self, x: Sequence[int]) -> np.ndarray:
        """The interpolated array at ``x`` (one index per ``x`` site)."""
        if len(x) != self.nsites:
            raise ValueError(f"Expected {self.nsites} indices, got {len(x)}.")
        cores = self.sitetensors()
        xcores = cores[:-1] if self.componentposition == "last" else cores[1:]
        return self._rescale(self._contract([T[:, int(i), :] for T, i in zip(xcores, x)]))

    def __call__(self, x: Sequence[int]) -> np.ndarray:
        return self.evaluate(x)

    def sum(self) -> np.ndarray:
        """Sum over the whole ``x`` index space, per component (linear cost)."""
        cores = self.sitetensors()
        xcores = cores[:-1] if self.componentposition == "last" else cores[1:]
        return self._rescale(self._contract([T.sum(axis=1) for T in xcores]))

    # -- per-component tensor trains ---------------------------------------

    def component(self, k) -> TensorTrain:
        """One component's own ``N``-site tensor train (an independent copy).

        ``k`` is either a flat index or a tuple index into ``valueshape``. The
        component site is sliced at ``k`` and absorbed into its neighbour, so
        this is *exact*: ``component(k)(x) == self(x).reshape(-1)[k]``. Every
        component shares all site tensors except the one adjacent to the
        component site.
        """
        idx = self._flatindex(k)
        cores = self.sitetensors()
        scale = 1.0 if self.componentweights is None else self.componentweights[idx]
        if self.componentposition == "last":
            merged = np.tensordot(cores[-2], cores[-1][:, idx, :], axes=([2], [0])) * scale
            tensors = [T.copy() for T in cores[:-2]] + [merged]
        else:
            merged = np.tensordot(cores[0][:, idx, :], cores[1], axes=([1], [0])) * scale
            tensors = [merged] + [T.copy() for T in cores[2:]]
        return TensorTrain(tensors)

    def components(self) -> list[TensorTrain]:
        return [self.component(k) for k in range(self.K)]


def crossinterpolate2_array(
    dtype,
    f: Callable,
    localdims: Sequence[int],
    valueshape,
    initialpivots: Sequence[Sequence[int]] | None = None,
    componentposition: str = "last",
    componentweights="auto",
    nsampleweights: int = 50,
    **kwargs,
) -> tuple[ArrayTensorTrain, list[int], list[float]]:
    """Cross interpolate an array-valued ``f(x) -> ndarray`` of shape ``valueshape``.

    All components are interpolated at once, sharing pivots, by running the
    ordinary TCI2 algorithm on the extended chain ``localdims + [K]`` (see the
    module docstring). ``**kwargs`` are forwarded to
    :func:`~qutecipy.tci2.optimize` unchanged (``tolerance``, ``maxbonddim``,
    ``pivotsearch``, ``maxiter``, ...).

    ``initialpivots`` entries may be given in ``x`` space (length
    ``len(localdims)``), in which case each is expanded into one extended pivot
    per component; extended pivots (length ``len(localdims) + 1``) are also
    accepted. The default is the all-zero ``x`` expanded over all components,
    so every component is sampled from the first iteration and the error
    normalization is honest immediately.

    ``componentweights`` defaults to ``"auto"``, which estimates each component's
    scale from ``nsampleweights`` sampled points and interpolates ``f_k / w_k``, so
    that a relative ``tolerance`` means the same thing for every component instead
    of being dominated by the largest one -- see "Error normalization" in the module
    docstring, and :func:`estimate_componentweights` for the estimator and its
    guards. Pass ``None`` for the unweighted behaviour (cheaper in bond dimension
    when the small components genuinely do not matter) or an explicit array of
    ``K`` nonzero weights. The sampled values are cached in the adapter TCI2 then
    uses, so ``"auto"`` does not cost ``nsampleweights`` extra evaluations of ``f``
    in the end -- only the ones TCI2 would not have made anyway.

    Returns ``(ArrayTensorTrain, ranks, errors)``. The underlying
    :class:`~qutecipy.tci2.TensorCI2` and the caching adapter are available as
    ``.tci`` and ``.func`` on the returned object; ``.componentweights`` holds the
    weights actually used.
    """
    autoweights = isinstance(componentweights, str)
    if autoweights and componentweights != "auto":
        raise ValueError(f"Unknown componentweights {componentweights!r}; use 'auto', None, or an array.")

    fa = ArrayValuedFunction(
        dtype, f, localdims, valueshape,
        componentposition=componentposition,
        componentweights=None if autoweights else componentweights,
    )

    if initialpivots is None:
        pivots = fa.extend_pivot([0] * fa.nsites)
    else:
        pivots = []
        for p in initialpivots:
            p = tuple(int(i) for i in p)
            if len(p) == fa.nsites:
                pivots.extend(fa.extend_pivot(p))
            elif len(p) == fa.nsites + 1:
                pivots.append(p)
            else:
                raise ValueError(
                    f"Initial pivot {p} has length {len(p)}; expected {fa.nsites} (x space) "
                    f"or {fa.nsites + 1} (extended)."
                )
        seen: set[tuple] = set()
        pivots = [p for p in pivots if not (p in seen or seen.add(p))]

    if autoweights:
        # Derive before the run, not after: TensorCI2.from_function seeds
        # maxsamplevalue from the initial pivots, and the whole point of the weights
        # is to make that normalization -- and hence `tolerance` -- mean the same
        # thing for every component. The initial pivots' own x values seed the
        # sample, so the point the user cared enough to name is always included.
        fa.set_componentweights(
            estimate_componentweights(
                fa, nsample=nsampleweights, xsamples=[fa.split(p)[0] for p in pivots]
            )
        )

    tci, ranks, errors = crossinterpolate2(dtype, fa, fa.extendedlocaldims, pivots, **kwargs)
    att = ArrayTensorTrain(
        tci, fa.valueshape, componentposition=componentposition, componentweights=fa.componentweights
    )
    att.tci = tci
    att.func = fa
    return att, ranks, errors
