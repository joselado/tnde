"""Batched function evaluation for TCI, with a switchable array backend.

``qutecipy.crossinterpolate2`` samples the function it interpolates through a
``BatchEvaluator``: given left index tuples, right index tuples and a number of free
centre sites, return the whole ``(nleft, 2, ..., 2, nright)`` block at once. Julia's
Gross-Pitaevskii code does not use that interface -- it wraps a *scalar* closure in
``TCI.CachedFunction`` and evaluates one point at a time -- so batching it was the
obvious place to expect a win from JAX.

Measured, it is not. See ``docs/backend_benchmarks.md``; the short version:

=========================  ==========  ==========  ==========  ======
workload (R=30)             Julia route  numpy       numpy+cache jax
=========================  ==========  ==========  ==========  ======
TCI of a Gaussian             0.09 s      0.12 s      --         1.25 s
one ``apply_f_tt`` step       1.70 s      2.57 s      0.74 s     7.05 s
=========================  ==========  ==========  ==========  ======

("Julia route" = a scalar closure in ``CachedFunction``, what the original does.)

TCI's batches are tiny -- 12 points when interpolating a plain function, 70 for
``apply_f_tt`` -- because the block size is set by the *rank*, which is 10-14 here.
At ~1700 kernel calls per Trotter step, JAX's per-dispatch floor dominates and no
amount of padding or jitting recovers it. Bulk evaluation does not rescue it either:
at 2**20 points in one call, JAX is still 2x slower than NumPy, because the cores are
14x14 and XLA's CPU backend has nothing to exploit at that size.

What *does* pay is batching in NumPy **combined with** memoising on the integer grid
index: TCI requests the same point many times over (122805 requests, 19067 distinct,
a 6.4x redundancy on the measurement above), so deduplicating is worth more than any
choice of array library. That combination is 2.3x faster than the Julia route.

So the default backend here is NumPy with caching, and JAX is kept as an option
rather than the default. JAX earns its place elsewhere in this port -- ``gptci.dense``, where the
arrays are full 2**R FFTs -- and would be worth revisiting for the tensor-train path
on a GPU, or at bond dimensions far above the 14 this problem uses.
"""
from __future__ import annotations

import numpy as np
from qutecipy.tensortrain.batcheval import BatchEvaluator

DEFAULT_BACKEND = "numpy"


def _next_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (int(n - 1).bit_length())


def _setvalues(indexset) -> np.ndarray:
    """Integer value of each index tuple, read MSB-first in base 2."""
    a = np.asarray(indexset, dtype=np.int64).reshape(len(indexset), -1)
    if a.shape[1] == 0:
        return np.zeros(len(indexset), dtype=np.int64)
    return a @ (1 << np.arange(a.shape[1] - 1, -1, -1, dtype=np.int64))


class QuanticsBatchFunction(BatchEvaluator):
    """Adapts a vectorised ``kernel(grid_indices) -> values`` to TCI's interface.

    Base 2 only, which is all the quantics representation needs.
    """

    def __init__(self, kernel, R: int, cache: bool = True):
        self.kernel = kernel
        self.R = R
        self._cache = {} if cache else None
        self.ncalls = 0
        self.npoints = 0
        self.nevaluated = 0

    def _call(self, idx: np.ndarray) -> np.ndarray:
        """Evaluate at ``idx``, reusing anything already computed.

        TCI's rook pivot search and global pivot finder revisit the same points
        repeatedly (qutecipy's own docs measure 5-15x redundancy), and a batch often
        repeats points internally too. Deduplicating with ``np.unique`` first and then
        consulting a dict keyed on the integer grid index gets both, and costs one
        vectorised sort per call.
        """
        self.ncalls += 1
        self.npoints += idx.size
        if self._cache is None:
            self.nevaluated += idx.size
            return self.kernel(idx)

        uniq, inv = np.unique(idx, return_inverse=True)
        out = np.empty(uniq.size, dtype=np.complex128)
        missing = np.empty(uniq.size, dtype=bool)
        get = self._cache.get
        for i, u in enumerate(uniq.tolist()):
            v = get(u)
            missing[i] = v is None
            if v is not None:
                out[i] = v
        if missing.any():
            todo = uniq[missing]
            vals = self.kernel(todo)
            out[missing] = vals
            self.nevaluated += todo.size
            self._cache.update(zip(todo.tolist(), vals.tolist()))
        return out[inv]

    def __call__(self, indexset) -> complex:
        return complex(self._call(_setvalues([indexset]))[0])

    def batchevaluate(self, leftindexset, rightindexset, ncent: int) -> np.ndarray:
        nl, nr = len(leftindexset), len(rightindexset)
        if nl * nr == 0:
            return np.empty((0,) * (ncent + 2), dtype=np.complex128)
        wl, wr = len(leftindexset[0]), len(rightindexset[0])
        if wl + ncent + wr != self.R:
            raise ValueError(f"widths {wl}+{ncent}+{wr} != R={self.R}")
        idx = ((_setvalues(leftindexset) << (ncent + wr))[:, None, None]
               + (np.arange(1 << ncent, dtype=np.int64) << wr)[None, :, None]
               + _setvalues(rightindexset)[None, None, :])
        return self._call(idx.ravel()).reshape((nl,) + (2,) * ncent + (nr,))


# --------------------------------------------------------------------------
# kernels
# --------------------------------------------------------------------------

def _x_of(idx, xmin, xmax, R):
    return xmin + idx.astype(np.float64) * ((xmax - xmin) / ((1 << R) - 1))


def _pad(cs, D):
    R = len(cs)
    out = np.zeros((R, D, 2, D), dtype=np.complex128)
    for n, c in enumerate(cs):
        out[n, : c.shape[0], :, : c.shape[2]] = c
    return out


def _tt_eval_numpy(pad, idx, R, D):
    """Evaluate a padded train at a batch of indices.

    Splitting the batch on the current bit and doing one dense matmul per branch keeps
    everything in BLAS; a per-element gather would not.
    """
    v = np.zeros((idx.size, D), dtype=np.complex128)
    v[:, 0] = 1.0
    for n in range(R):
        b = ((idx >> (R - 1 - n)) & 1).astype(bool)
        out = np.empty_like(v)
        nb = ~b
        if nb.any():
            out[nb] = v[nb] @ pad[n, :, 0, :]
        if b.any():
            out[b] = v[b] @ pad[n, :, 1, :]
        v = out
    return v[:, 0]


def _jax_wrap(fn):
    """Wrap a jnp-written kernel so it sees only power-of-two batch sizes, capping the
    number of XLA retraces. Only used when backend='jax'."""
    import jax
    import jax.numpy as jnp

    jitted = jax.jit(fn)

    def call(idx):
        n = idx.shape[0]
        if n == 0:
            return np.zeros(0, dtype=np.complex128)
        m = _next_pow2(n)
        if m != n:
            idx = np.concatenate([idx, np.zeros(m - n, dtype=idx.dtype)])
        return np.asarray(jitted(jnp.asarray(idx)))[:n]

    return call


# --------------------------------------------------------------------------
# constructors
# --------------------------------------------------------------------------

def grid_function(fn, R: int, xmin: float, xmax: float,
                  backend: str = DEFAULT_BACKEND, cache: bool = True) -> QuanticsBatchFunction:
    """TCI-ready evaluator for ``fn(x)`` on the ``includeendpoint`` grid.

    With ``backend='numpy'`` (the default) ``fn`` must be vectorised over a NumPy
    array; with ``backend='jax'`` it must be written against ``jax.numpy``.
    """
    if backend == "numpy":
        def kernel(idx):
            return np.asarray(fn(_x_of(idx, xmin, xmax, R)), dtype=np.complex128)
    elif backend == "jax":
        import jax.numpy as jnp
        dx = (xmax - xmin) / ((1 << R) - 1)
        kernel = _jax_wrap(lambda i: jnp.asarray(
            fn(xmin + i.astype(jnp.float64) * dx), dtype=jnp.complex128))
    else:
        raise ValueError(f"unknown backend {backend!r}")
    return QuanticsBatchFunction(kernel, R, cache=cache)


def tt_function(psi, fn, R: int, xmin: float, xmax: float, D: int | None = None,
                backend: str = DEFAULT_BACKEND, cache: bool = True) -> QuanticsBatchFunction:
    """TCI-ready evaluator for ``fn(psi(x), x)`` -- the ``apply_f_tt`` pattern, where
    the interpolated function is built from the current wave function.

    ``D`` is the bond dimension the cores are padded to; pass the run's ``maxdim`` to
    keep the shape (and, under JAX, the compiled kernel) constant across steps.
    """
    from gptci import tt as _tt

    cs = _tt.cores(psi)
    need = max(max(c.shape[0] for c in cs), max(c.shape[2] for c in cs))
    if D is None:
        D = need
    elif D < need:
        raise ValueError(f"D={D} < the state's bond dimension {need}")
    pad = _pad(cs, D)

    if backend == "numpy":
        def kernel(idx):
            wf = _tt_eval_numpy(pad, idx, R, D)
            return np.asarray(fn(wf, _x_of(idx, xmin, xmax, R)), dtype=np.complex128)
    elif backend == "jax":
        import jax
        import jax.numpy as jnp
        padj = jnp.asarray(pad)
        dx = (xmax - xmin) / ((1 << R) - 1)
        shifts = jnp.arange(R - 1, -1, -1)

        def raw(idx):
            bits = (idx[:, None] >> shifts[None, :]) & 1
            v = jnp.zeros((idx.size, D), jnp.complex128).at[:, 0].set(1.0)

            def step(v, n):
                b = bits[:, n][:, None]
                return jnp.where(b == 0, v @ padj[n, :, 0, :], v @ padj[n, :, 1, :]), None

            v, _ = jax.lax.scan(step, v, jnp.arange(R))
            x = xmin + idx.astype(jnp.float64) * dx
            return jnp.asarray(fn(v[:, 0], x), dtype=jnp.complex128)

        kernel = _jax_wrap(raw)
    else:
        raise ValueError(f"unknown backend {backend!r}")
    return QuanticsBatchFunction(kernel, R, cache=cache)


# --------------------------------------------------------------------------
# 2D: interleaved unfolding
# --------------------------------------------------------------------------

def interleave(ix, iy, R: int) -> np.ndarray:
    """Fuse two R-bit grid indices into the 2R-bit index of an interleaved train.

    ``QuanticsGrids``' ``unfoldingscheme=:interleaved`` orders the digits
    ``x_1 y_1 x_2 y_2 ... x_R y_R`` with ``x_1`` the most significant bit of ``x``, so
    the two variables share length scales site by site -- which is what keeps a 2D
    function's bond dimension manageable.
    """
    ix = np.asarray(ix, dtype=np.int64)
    iy = np.asarray(iy, dtype=np.int64)
    out = np.zeros(np.broadcast(ix, iy).shape, dtype=np.int64)
    for n in range(R):
        out |= ((ix >> (R - 1 - n)) & 1) << (2 * R - 1 - 2 * n)
        out |= ((iy >> (R - 1 - n)) & 1) << (2 * R - 2 - 2 * n)
    return out


def deinterleave(idx, R: int):
    """Inverse of :func:`interleave`."""
    idx = np.asarray(idx, dtype=np.int64)
    ix = np.zeros_like(idx)
    iy = np.zeros_like(idx)
    for n in range(R):
        ix |= ((idx >> (2 * R - 1 - 2 * n)) & 1) << (R - 1 - n)
        iy |= ((idx >> (2 * R - 2 - 2 * n)) & 1) << (R - 1 - n)
    return ix, iy


def _xy_of(idx, R, xmin, xmax, ymin, ymax, includeendpoint=False):
    """Coordinates of an interleaved index.

    ``includeendpoint`` defaults to False here because ``GP_2D.jl`` builds its grid
    *without* it, unlike the 1D code -- so the 2D spacing is ``L/2**R`` and agrees with
    the ``dx`` used by ``normalise_2D``, where the 1D pair disagree. Do not unify them.
    """
    den = ((1 << R) - 1) if includeendpoint else (1 << R)
    ix, iy = deinterleave(idx, R)
    return (xmin + ix * ((xmax - xmin) / den),
            ymin + iy * ((ymax - ymin) / den))


def grid_function_2d(fn, R, xmin, xmax, ymin, ymax, cache=True,
                     includeendpoint=False) -> QuanticsBatchFunction:
    """TCI-ready evaluator for ``fn(x, y)`` on an interleaved 2D quantics grid."""
    def kernel(idx):
        x, y = _xy_of(idx, R, xmin, xmax, ymin, ymax, includeendpoint)
        return np.asarray(fn(x, y), dtype=np.complex128)

    return QuanticsBatchFunction(kernel, 2 * R, cache=cache)


def tt_function_2d(psi, fn, R, xmin, xmax, ymin, ymax, D=None, cache=True,
                   includeendpoint=False) -> QuanticsBatchFunction:
    """TCI-ready evaluator for ``fn(psi(x,y), x, y)`` -- ``apply_f_tt_2D``."""
    from gptci import tt as _tt

    cs = _tt.cores(psi)
    need = max(max(c.shape[0] for c in cs), max(c.shape[2] for c in cs))
    D = need if D is None else max(D, need)
    pad = _pad(cs, D)
    n_sites = len(cs)

    def kernel(idx):
        wf = _tt_eval_numpy(pad, idx, n_sites, D)
        x, y = _xy_of(idx, R, xmin, xmax, ymin, ymax, includeendpoint)
        return np.asarray(fn(wf, x, y), dtype=np.complex128)

    return QuanticsBatchFunction(kernel, n_sites, cache=cache)
