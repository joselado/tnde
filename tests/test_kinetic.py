"""Gate for the kinetic propagator.

This is the single most convention-dense object in the port: it composes the quantics
Fourier MPO (whose site order is inverted), a diagonal momentum-space operator built
on a *different* grid, and the inverse transform, with three ``reverse`` calls that
each fail silently if wrong. Checking it against the dense FFT solver validates the
Fourier sign, the unitary normalisation, all three reversals, the Fermi-Dirac
low-pass and the finite-difference dispersion in one shot.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from qutecipy import contract, crossinterpolate2

from tnde import batcheval, dense, tt
from tnde.operators import kinetic_mpo_at

R, XMIN, XMAX, DT, M, BETA, TOL = 12, -40.0, 40.0, 0.05, 1.0, 2.0, 1e-12


def _state():
    psi0 = lambda x: (np.exp(-((x - 1.5) ** 2) / 2) * (1 + 0.2 * np.cos(3 * x))).astype(np.complex128)
    f = batcheval.grid_function(psi0, R, XMIN, XMAX)
    p = [0] * R
    p[0] = 1                                  # pivot at the grid centre, near the peak
    ci, _, _ = crossinterpolate2(np.complex128, f, [2] * R, [p], tolerance=TOL)
    return tt.from_cores(tt.cores(ci))


def test_matches_dense_fft():
    psi = _state()
    d0 = tt.to_dense(psi)
    for kexp in (5, 7, 9):
        op = kinetic_mpo_at(R, XMIN, XMAX, DT, M, 2**kexp, TOL)
        got = tt.to_dense(contract(op, psi, algorithm="naive", tolerance=TOL))
        kfac = dense.kinetic_factor(R, XMIN, XMAX, DT, M, 2**kexp, BETA)
        want = np.asarray(dense.kinetic_step(d0, kfac))
        e = np.max(np.abs(got - want)) / np.max(np.abs(want))
        print(f"  kcut = {2**kexp:>4}: operator rank {op.rank():>3}, "
              f"||TT - dense|| / ||dense|| = {e:.3e}")
        assert e < 1e-10


if __name__ == "__main__":
    test_matches_dense_fft()
    print("PASS")
