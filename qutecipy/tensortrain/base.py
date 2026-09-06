"""Port of abstracttensortrain.jl: the AbstractTensorTrain contract.

``add``/``subtract`` (which need TensorTrain + compress) live in core.py to
avoid a circular import; everything else that only needs ``sitetensors()``
lives here as mixin methods, matching the Julia file's structure.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class AbstractTensorTrain(ABC):
    @abstractmethod
    def sitetensors(self) -> list[np.ndarray]: ...

    def sitetensor(self, i: int) -> np.ndarray:
        return self.sitetensors()[i]

    def linkdims(self) -> list[int]:
        return [T.shape[0] for T in self.sitetensors()[1:]]

    def linkdim(self, i: int) -> int:
        return self.sitetensor(i + 1).shape[0]

    def sitedims(self) -> list[list[int]]:
        return [list(T.shape[1:-1]) for T in self.sitetensors()]

    def sitedim(self, i: int) -> list[int]:
        return list(self.sitetensor(i).shape[1:-1])

    def rank(self) -> int:
        linkdims = self.linkdims()
        return max(linkdims) if linkdims else 0

    def __len__(self) -> int:
        return len(self.sitetensors())

    def __iter__(self):
        return iter(self.sitetensors())

    def __getitem__(self, i):
        return self.sitetensor(i)

    def evaluate(self, indexset):
        """Evaluate the tensor train at indexset: one local index per site
        (a scalar, or -- for tensors with fused/multiple physical legs -- a
        sequence of indices) via a chain of matrix products."""
        sts = self.sitetensors()
        if len(indexset) != len(sts):
            raise ValueError(
                f"To evaluate a tt of length {len(sts)}, you have to provide "
                f"{len(sts)} indices, but there were {len(indexset)}."
            )
        # The chain is carried as a 1-D vector rather than a 1 x r matrix: the products
        # are then BLAS gemv instead of a degenerate gemm, and the leading `1` axis never
        # has to be re-sliced. Same operands, same left-to-right association, same
        # summation order -- bit-identical results, just less per-site overhead.
        result = None
        for T, i in zip(sts, indexset):
            if isinstance(i, (list, tuple, np.ndarray)):
                if len(i) != T.ndim - 2:
                    raise ValueError(f"Index {i} does not match tensor shape {T.shape}.")
                mat = T[(slice(None), *i, slice(None))]
            else:
                mat = T[:, i, :]
            result = mat[0] if result is None else result @ mat
        return result[0]

    def __call__(self, indexset):
        return self.evaluate(indexset)

    def sum(self, dims: tuple[int, ...] | int | None = None):
        if dims is None:
            dims = tuple(range(len(self)))
        elif isinstance(dims, int):
            dims = (dims,)
        return _sum(self, dims=dims)

    def norm2(self) -> float:
        """``<tt|tt>``, carried as a bond x bond environment rather than a transfer matrix.

        Forming each site's transfer matrix explicitly -- ``tensordot(conj(t), t)`` over
        the physical leg, giving ``(chi, chi, chi, chi)`` -- costs ``chi**4`` in memory
        and time, which for a train that has not been compressed is fatal rather than
        slow: a rank-275 core (the uncompressed product of a rank-25 operator and a
        rank-11 state) asks for 91 GB. Contracting the environment through each core
        instead is the same quantity in ``chi**2`` memory and ``chi**3 * d`` time.
        """
        env = np.ones((1, 1))
        for n in range(len(self)):
            t = self.sitetensor(n)
            t3 = t.reshape(t.shape[0], -1, t.shape[-1])
            env = np.tensordot(env, t3, axes=([1], [0]))            # (lc, s, r)
            env = np.tensordot(np.conj(t3), env, axes=([0, 1], [0, 1]))  # (rc, r)
        return float(np.real(env.reshape(-1)[0]))

    def norm(self) -> float:
        return float(np.sqrt(self.norm2()))


def _sum(tt: AbstractTensorTrain, dims: tuple[int, ...]):
    from qutecipy.tensortrain.core import TensorTrain

    sts = tt.sitetensors()
    dtype = sts[0].dtype
    tensors: list[np.ndarray] = []
    tprod = np.eye(1, dtype=dtype)
    for n, T in enumerate(sts):
        if n in dims:
            tprod = tprod @ np.sum(T, axis=1)
        else:
            tprod = tprod @ T.reshape(T.shape[0], -1)
            tensors.append(tprod.reshape(tprod.shape[0], T.shape[1], T.shape[-1]))
            tprod = np.eye(T.shape[-1], dtype=dtype)
    if tensors:
        last = tensors[-1]
        tprod2 = last.reshape(-1, last.shape[2]) @ tprod
        tensors[-1] = tprod2.reshape(last.shape[0], last.shape[1], tprod2.shape[1])
        return TensorTrain(tensors)
    return tprod.reshape(-1)[0]
