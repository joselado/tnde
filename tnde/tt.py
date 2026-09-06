"""Tensor-train / MPS utilities shared by the 1D and 2D solvers.

Everything here operates on ``qutecipy.TensorTrain`` objects, whose site tensors are
NumPy arrays in the same ``(left, physical..., right)`` layout Julia's
``TCI.TensorTrain`` uses -- so the reference cores dumped from the paper's ``.jld2``
files can be fed in directly.

Index conventions (see PORTING_NOTES.md): qutecipy is 0-based in both grid index and
quantics digit where Julia is 1-based, but the *bitstring to coordinate* map is
identical in the two, so no translation is needed as long as one stays in bitstrings.
The single place the offset bites is a pivot copied literally out of the Julia source:
``[2,1,1,...]`` there is ``[1,0,0,...]`` here.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from qutecipy import TensorTrain

from tnde import config  # noqa: F401  (enables x64 on import)


# --------------------------------------------------------------------------
# core extraction / construction
# --------------------------------------------------------------------------

def cores(obj) -> list[np.ndarray]:
    """The site tensors of a TT-like object (``TensorTrain``, ``TensorCI2``, ...) as
    3-leg ``(left, phys, right)`` arrays."""
    out = []
    for n in range(len(obj)):
        t = np.asarray(obj.sitetensor(n))
        out.append(t.reshape(t.shape[0], -1, t.shape[-1]))
    return out


def from_cores(cs) -> TensorTrain:
    return TensorTrain([np.ascontiguousarray(c) for c in cs])


def bonddims(obj) -> list[int]:
    cs = cores(obj)
    return [c.shape[0] for c in cs] + [cs[-1].shape[-1]]


# --------------------------------------------------------------------------
# TT -> MPO
# --------------------------------------------------------------------------

def tt_to_mpo(obj) -> TensorTrain:
    """Embed a function-valued TT as a *diagonal* MPO, so that contracting it with a
    state performs elementwise multiplication (Julia's ``TT_to_MPO``).

    ``M[a,e,f,c] = t[a,b,c] delta[e,f,b]`` -- i.e. the physical leg is copied onto the
    diagonal of an (out, in) pair.
    """
    out = []
    for t in cores(obj):
        l, d, r = t.shape
        M = np.zeros((l, d, d, r), dtype=np.result_type(t.dtype, np.complex128))
        for b in range(d):
            M[:, b, b, :] = t[:, b, :]
        out.append(M)
    return TensorTrain(out)


# --------------------------------------------------------------------------
# embedding an MPO into a subset of sites
# --------------------------------------------------------------------------

def embed_mpo(cores, positions, nsites) -> TensorTrain:
    """Make an ``len(cores)``-site MPO act on ``positions`` of an ``nsites`` chain.

    The untouched sites get an identity core that also carries the operator's bond
    through unchanged -- ``I[a, s, s', b] = delta_{ss'} delta_{ab}`` -- so the embedded
    operator has exactly the original's bond dimension. This is the plain-array
    equivalent of ITensor's ``matchsiteinds``.
    """
    positions = set(int(p) for p in positions)
    out, bond, ci = [], 1, 0
    for p in range(nsites):
        if p in positions:
            c = np.asarray(cores[ci])
            out.append(c)
            bond = c.shape[-1]
            ci += 1
        else:
            I = np.zeros((bond, 2, 2, bond), dtype=np.complex128)
            idx = np.arange(bond)
            I[idx, 0, 0, idx] = 1.0
            I[idx, 1, 1, idx] = 1.0
            out.append(I)
    if ci != len(cores):
        raise ValueError(f"consumed {ci} of {len(cores)} cores")
    return TensorTrain(out)


# --------------------------------------------------------------------------
# overlaps, norms, normalisation
# --------------------------------------------------------------------------

def inner(a, b) -> complex:
    """``<a|b>`` -- the plain tensor-train overlap, no measure factor."""
    E = np.ones((1, 1), dtype=np.complex128)
    for A, B in zip(cores(a), cores(b)):
        E = np.einsum("lm,lpi,mpj->ij", E, A.conj(), B, optimize=True)
    return complex(E[0, 0])


def fidelity(a, b, R: int, xmin: float, xmax: float) -> float:
    """``|<a|b> dx|**2`` with ``dx = L/2**R`` (Julia's ``fidelity_ITensor``).

    Note this is a *squared* overlap, and that the ``dx`` here is deliberately not the
    ``includeendpoint`` grid spacing ``L/(2**R - 1)``. Both quirks are in the original
    and are preserved so the numbers match.
    """
    dx = (xmax - xmin) / (1 << R)
    return float(abs(inner(a, b) * dx) ** 2)


def normalise(obj, R: int, xmin: float, xmax: float) -> TensorTrain:
    """Rescale to unit norm by spreading ``1/n`` over the R cores as ``n**(-1/R)``
    each, where ``n = fidelity**(1/4)`` (Julia's ``normalise``).

    Spreading the factor rather than scaling one core keeps the cores' magnitudes
    comparable, which matters at R = 30 where ``n**(1/R)`` is very close to 1.
    """
    cs = cores(obj)
    n = np.sqrt(np.sqrt(fidelity(obj, obj, R, xmin, xmax)))
    n_R = n ** (1.0 / len(cs))
    return from_cores([c / n_R for c in cs])


# --------------------------------------------------------------------------
# batched evaluation (JAX)
# --------------------------------------------------------------------------

def pad_cores(cs, D: int) -> jnp.ndarray:
    """Stack cores into one ``(R, D, 2, D)`` array, zero-padded to a common bond
    dimension ``D``.

    Padding is what lets ``jax.jit`` be used at all here: the bond dimension changes
    from step to step and from core to core, and every distinct shape would otherwise
    trigger a recompile. It is exact, not an approximation -- the boundary vector has
    support only on the first entry, and each padded core maps the first ``l`` rows to
    the first ``r`` columns with zeros elsewhere, so the zeros never mix in.
    """
    R = len(cs)
    out = np.zeros((R, D, 2, D), dtype=np.complex128)
    for n, c in enumerate(cs):
        out[n, : c.shape[0], :, : c.shape[2]] = c
    return jnp.asarray(out)


def _eval_numpy(padded: np.ndarray, R: int, idx: np.ndarray) -> np.ndarray:
    """Evaluate a padded train at a batch of 0-based grid indices, MSB-first.

    Splitting the batch on the current bit and doing one dense matmul per branch keeps
    the work in BLAS; a per-element gather would not.
    """
    D = padded.shape[1]
    v = np.zeros((idx.size, D), dtype=np.complex128)
    v[:, 0] = 1.0
    for n in range(R):
        b = ((idx >> (R - 1 - n)) & 1).astype(bool)
        out = np.empty_like(v)
        nb = ~b
        if nb.any():
            out[nb] = v[nb] @ padded[n, :, 0, :]
        if b.any():
            out[b] = v[b] @ padded[n, :, 1, :]
        v = out
    return v[:, 0]


@partial(jax.jit, static_argnums=(1,))
def _eval_jax(padded: jnp.ndarray, R: int, idx: jnp.ndarray) -> jnp.ndarray:
    """JAX equivalent of :func:`_eval_numpy`. Both branches are computed and blended
    rather than gathered, so the body stays a dense matmul XLA can fuse."""
    D = padded.shape[1]
    shifts = jnp.arange(R - 1, -1, -1)
    bits = (idx[:, None] >> shifts[None, :]) & 1
    v = jnp.zeros((idx.size, D), jnp.complex128).at[:, 0].set(1.0)

    def step(v, n):
        b = bits[:, n][:, None]
        return jnp.where(b == 0, v @ padded[n, :, 0, :], v @ padded[n, :, 1, :]), None

    v, _ = jax.lax.scan(step, v, jnp.arange(R))
    return v[:, 0]


def evaluate(obj, idx, D: int | None = None, backend: str = "numpy") -> np.ndarray:
    """Evaluate a TT at an array of 0-based grid indices.

    NumPy is the default because it is the faster one here, by a factor of two even at
    2**20 points in a single call: the cores are at most 14x14, which is far too small
    for XLA's CPU backend to exploit. The JAX path is kept for a GPU, or for bond
    dimensions much larger than this problem's. See ``tnde.batcheval`` for the
    measurements.
    """
    cs = cores(obj)
    if D is None:
        D = max(max(c.shape[0] for c in cs), max(c.shape[2] for c in cs))
    idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
    padded = np.zeros((len(cs), D, 2, D), dtype=np.complex128)
    for n, c in enumerate(cs):
        padded[n, : c.shape[0], :, : c.shape[2]] = c
    if backend == "numpy":
        return _eval_numpy(padded, len(cs), idx)
    if backend == "jax":
        return np.asarray(_eval_jax(jnp.asarray(padded), len(cs), jnp.asarray(idx)))
    raise ValueError(f"unknown backend {backend!r}")


def to_dense(obj) -> np.ndarray:
    """The full 2**R vector. Only for small R -- used by the tests as ground truth."""
    R = len(cores(obj))
    return evaluate(obj, np.arange(1 << R))


def max_error(obj, fn, R: int, xmin: float, xmax: float, idx=None,
              npoints: int = 20001, halfwidth: float | None = None) -> float:
    """True maximum relative error of a train against the function it represents.

    TCI's own reported error is estimated *on its own pivots*, so it says nothing about
    regions the pivots never reached: on the paper's initial state it reports 7e-11
    while missing half the Gaussian entirely (true error 1.0). Any TCI result that
    matters should be checked against an independent sample, which is what this does.

    ``halfwidth`` restricts the sample to ``|x| <= halfwidth`` -- necessary on a wide
    box, where a uniform sample lands almost entirely in the region where the function
    is numerically zero and the comparison is dominated by noise.
    """
    dx = (xmax - xmin) / ((1 << R) - 1)
    if idx is None:
        if halfwidth is None:
            idx = np.round(np.linspace(0, (1 << R) - 1, npoints)).astype(np.int64)
        else:
            c = 1 << (R - 1)
            w = int(halfwidth / dx)
            idx = c + np.round(np.linspace(-w, w, npoints)).astype(np.int64)
            idx = np.clip(idx, 0, (1 << R) - 1)
    exact = np.asarray(fn(xmin + idx * dx), dtype=np.complex128)
    got = evaluate(obj, idx)
    scale = np.max(np.abs(exact))
    return float(np.max(np.abs(got - exact)) / scale) if scale > 0 else float(np.max(np.abs(got)))
