"""Variational (fit) application of an MPO to an MPS.

``ITensors.contract(mpo, mps; method="fit", nsweeps, maxdim)`` -- what the original
uses for every MPO application -- never forms the exact product. It sweeps, projecting
``W|psi>`` into the environment of a bond-limited ansatz and re-solving site by site.

The alternative, ``contract(algorithm="naive")``, builds the full product and then
compresses it globally. In 1D that is fine and in fact slightly *more* accurate
(measured 7.4e-06 against the exact product at maxdim=14, versus zip-up's 5.1e-02), so
the 1D driver uses it. In 2D it is not: the momentum-space kinetic MPO has rank ~65 and
the state runs at maxdim 50, so the exact intermediate carries bond dimensions near
3000 and a single application costs 45-240 s. This module is what makes the 2D path
practical.

Two-site sweeps are used rather than single-site, so the bond dimension adapts up to
``maxdim`` from whatever the initial guess had; a single-site fit can never grow past
its starting rank.
"""
from __future__ import annotations

import numpy as np
from qutecipy.tensortrain.core import TensorTrain

from gptci import tt


def _canonicalise_right(cores):
    """Make every core but the first right-orthogonal, via RQ from the right."""
    c = [np.array(x, dtype=np.complex128) for x in cores]
    for n in range(len(c) - 1, 0, -1):
        t = c[n]
        left, rest = t.shape[0], int(np.prod(t.shape[1:]))
        q, r = np.linalg.qr(t.reshape(left, rest).conj().T)
        k = q.shape[1]
        c[n] = q.conj().T.reshape((k,) + t.shape[1:])
        c[n - 1] = np.tensordot(c[n - 1], r.conj().T, axes=([c[n - 1].ndim - 1], [0]))
    return c


def _truncate(cores, maxdim):
    t = TensorTrain([np.array(x) for x in cores])
    t.compress("SVD", tolerance=0.0, maxbonddim=maxdim)
    return [np.asarray(t.sitetensor(n)) for n in range(len(t))]


# Every contraction below is written as an explicit chain of `tensordot` calls rather
# than one `einsum`. This is not stylistic: `np.einsum(..., optimize=True)` picks a
# catastrophic path for the four-operand environment updates -- a single `_renv` at
# 2D sizes (chi_phi 50, chi_W 63, chi_psi 43) did not finish in 300 s, against ~30 ms
# for the staged version. Each step below is a genuine matrix multiply, so the cost is
# the O(chi_phi * chi_W * chi_psi * chi_W * 4) it should be.

def _renv(phi, w, p, nxt):
    """R[c,b,a] = sum conj(phi)[c,S,x] w[b,S,T,f] p[a,T,z] nxt[x,f,z]."""
    t = np.tensordot(phi.conj(), nxt, axes=([2], [0]))       # (c,S,f,z)
    t = np.tensordot(t, p, axes=([3], [2]))                  # (c,S,f,a,T)
    t = np.tensordot(t, w, axes=([1, 4, 2], [1, 2, 3]))      # (c,a,b)
    return t.transpose(0, 2, 1)


def _lenv(phi, w, p, prev):
    """L[c,b,a] = sum conj(phi)[x,S,c] w[f,S,T,b] p[z,T,a] prev[x,f,z]."""
    t = np.tensordot(phi.conj(), prev, axes=([0], [0]))      # (S,c,f,z)
    t = np.tensordot(t, p, axes=([3], [0]))                  # (S,c,f,T,a)
    t = np.tensordot(t, w, axes=([0, 2, 3], [1, 0, 2]))      # (c,a,b)
    return t.transpose(0, 2, 1)


def _theta(L, w1, p1, w2, p2, R):
    """The two-site block of ``W|psi>`` projected into the current environments.

    Assembled from both ends so the largest intermediate stays
    ``O(maxdim * chi_W * chi_psi * 2)``; forming the product first is exactly what
    makes the naive route expensive.
    """
    A = np.tensordot(L, p1, axes=([2], [0]))                 # (c,f,T,u)
    A = np.tensordot(A, w1, axes=([1, 2], [0, 2]))           # (c,u,S,g)
    A = A.transpose(0, 2, 3, 1)                              # (c,S,g,u)
    B = np.tensordot(R, p2, axes=([2], [2]))                 # (d,h,u,T)
    B = np.tensordot(B, w2, axes=([1, 3], [3, 2]))           # (d,u,g,S)
    B = B.transpose(0, 3, 2, 1)                              # (d,S,g,u)
    return np.tensordot(A, B, axes=([2, 3], [2, 3])).transpose(0, 1, 3, 2)


def _svd(mat):
    try:
        return np.linalg.svd(mat, full_matrices=False)
    except np.linalg.LinAlgError:                     # divide-and-conquer can fail
        import scipy.linalg
        return scipy.linalg.svd(mat, full_matrices=False, lapack_driver="gesvd")


def _split(theta, maxdim, tolerance, centre="right"):
    """Split a two-site block, leaving the orthogonality centre on the named side.

    Which side matters: the two-site projection is only the correct one when the
    environments are orthonormal, which requires everything left of the active bond to
    be left-orthogonal and everything right of it right-orthogonal. A sweep that always
    puts the centre on the right silently breaks that on the way back, and the
    environments then grow without bound -- in 2D they overflow to inf within one
    sweep.
    """
    c, s1, s2, d = theta.shape
    u, sv, vh = _svd(theta.reshape(c * s1, s2 * d))
    keep = int(np.sum(sv > tolerance * (sv[0] if sv.size else 1.0)))
    keep = max(1, min(keep, maxdim, sv.size))
    u, sv, vh = u[:, :keep], sv[:keep], vh[:keep]
    if centre == "right":
        return u.reshape(c, s1, keep), (sv[:, None] * vh).reshape(keep, s2, d)
    return (u * sv[None, :]).reshape(c, s1, keep), vh.reshape(keep, s2, d)


def apply_mpo(W, psi, maxdim, nsweeps=2, tolerance=1e-12, x0=None):
    """``W |psi>`` as an MPS of bond dimension at most ``maxdim``.

    ``x0`` is the starting ansatz; by default the state itself, truncated. That is a
    good guess here because most of the operators applied are diagonal in real space,
    so the product is a pointwise reweighting of ``psi`` and shares its correlation
    structure.
    """
    w = [np.asarray(W.sitetensor(n)) for n in range(len(W))]
    p = tt.cores(psi)
    N = len(p)
    if len(w) != N:
        raise ValueError(f"MPO has {len(w)} sites, state has {N}")

    guess = tt.cores(x0) if x0 is not None else p
    phi = _canonicalise_right(_truncate(guess, maxdim))

    one = np.ones((1, 1, 1), dtype=np.complex128)
    R = [None] * (N + 1)
    R[N] = one
    for n in range(N - 1, -1, -1):
        R[n] = _renv(phi[n], w[n], p[n], R[n + 1])
    L = [None] * (N + 1)
    L[0] = one

    for _ in range(nsweeps):
        for n in range(N - 1):                      # left to right
            th = _theta(L[n], w[n], p[n], w[n + 1], p[n + 1], R[n + 2])
            phi[n], phi[n + 1] = _split(th, maxdim, tolerance, centre="right")
            L[n + 1] = _lenv(phi[n], w[n], p[n], L[n])
        for n in range(N - 2, -1, -1):              # right to left
            th = _theta(L[n], w[n], p[n], w[n + 1], p[n + 1], R[n + 2])
            phi[n], phi[n + 1] = _split(th, maxdim, tolerance, centre="left")
            R[n + 1] = _renv(phi[n + 1], w[n + 1], p[n + 1], R[n + 2])

    return TensorTrain(phi)
