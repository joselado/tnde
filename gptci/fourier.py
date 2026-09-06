"""The quantics Fourier transform as an MPO.

Port of ``QuanticsTCI.jl`` v0.7.0 ``src/fouriertransform.jl`` -- the one piece of the
Julia stack that ``qutecipy`` does not yet cover. The construction is the direct
interpolative one of Chen and Lindsey (arXiv:2404.03182): the DFT kernel
``exp(2 pi i s k m / M)`` is written analytically as a tensor train whose bond index
runs over a Chebyshev-Lagrange basis of size ``K+1``, and the result is then
recompressed to ``maxbonddim``. Because it is analytic there is no interpolation
sweep and no function sampling -- it is exact up to the final compression.

Index ordering (this is the trap; see PORTING_NOTES.md)
------------------------------------------------------
Going in, the leftmost site is ``sigma_1``, the *largest* length scale. Coming out,
the transformed train is ordered the other way round -- leftmost is ``sigma'_R``, the
*smallest*. That inversion is intrinsic to the QFT and is what keeps the bond
dimension small; it is not an implementation detail that can be tidied away.
``qutecipy.reverse`` restores the usual large-to-small order, and every use of these
operators has to account for it. ``kinetic_mpo`` below does exactly that.
"""
from __future__ import annotations

import numpy as np
from qutecipy import TensorTrain


def _chebyshev_lagrange(K: int):
    """Chebyshev nodes on [0,1] plus their barycentric weights.

    ``grid[j] = (1 - cos(pi j / K)) / 2`` for ``j = 0..K``.
    """
    j = np.arange(K + 1)
    grid = 0.5 * (1.0 - np.cos(np.pi * j / K))
    w = np.empty(K + 1, dtype=np.float64)
    for a in range(K + 1):
        d = grid[a] - grid
        d[a] = 1.0
        w[a] = 1.0 / np.prod(d)
    return grid, w


def _lagrange_values(grid: np.ndarray, w: np.ndarray, x: np.ndarray) -> np.ndarray:
    """``L[a, i] = l_a(x_i)``, the barycentric Lagrange basis.

    ``l_a(x) = prod_m (x - grid[m]) * w[a] / (x - grid[a])``, with the removable
    singularity at a node handled explicitly (Julia uses a 1e-14 guard).
    """
    diff = x[None, :] - grid[:, None]                      # (K+1, nx)
    node_poly = np.prod(diff, axis=0)                      # prod_m (x - grid[m])
    with np.errstate(divide="ignore", invalid="ignore"):
        L = node_poly[None, :] * w[:, None] / diff
    at_node = np.abs(diff) < 1e-14
    L[at_node] = 1.0
    return L


def fourier_mpo(R: int, sign: float = -1.0, tolerance: float = 1e-14,
                maxbonddim: int = 12, K: int = 25, method: str = "SVD",
                normalize: bool = True) -> TensorTrain:
    """The R-site quantics DFT operator.

    Contracted with a quantics train ``F``, it produces
    ``F~_k = sum_m F_m exp(2 pi i * sign * k m / M)``, ``M = 2**R``, with the output
    train in *reversed* site order (see the module docstring).

    ``normalize=True`` divides every core by ``sqrt(2)``, i.e. the whole operator by
    ``sqrt(M)``, making it the unitary DFT -- so a forward/inverse pair composes to
    the identity with no leftover factor.

    ``K`` is the size of the interpolation basis *before* compression; the Julia
    default of 25 is kept, and the docs there warn it becomes inaccurate below 22.
    """
    if R < 2:
        raise ValueError("R must be at least 2.")
    grid, w = _chebyshev_lagrange(K)

    # A[alpha, tau, sigma, beta] = l_alpha(x) * exp(2 pi i * sign * x * tau),
    # with x = (sigma + grid[beta]) / 2.
    sigma = np.array([0, 1])
    x = (sigma[:, None] + grid[None, :]) / 2.0            # (2, K+1), indexed [sigma, beta]
    L = _lagrange_values(grid, w, x.ravel()).reshape(K + 1, 2, K + 1)   # [alpha, sigma, beta]
    tau = np.array([0, 1])
    phase = np.exp(2j * np.pi * sign * x[None, :, :] * tau[:, None, None])  # [tau, sigma, beta]
    A = np.einsum("asb,tsb->atsb", L, phase)              # (K+1, 2, 2, K+1)

    first = A.sum(axis=0).reshape(1, 2, 2, K + 1)
    last = A[:, :, :, 0].reshape(K + 1, 2, 2, 1)
    tt = TensorTrain([first] + [A.copy() for _ in range(R - 2)] + [last])
    tt.compress(method, tolerance=tolerance, maxbonddim=maxbonddim)

    if normalize:
        for n in range(len(tt)):
            tt.sitetensor(n)[...] /= np.sqrt(2.0)
    return tt
