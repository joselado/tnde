"""End-to-end gate for the 1D solver: the tensor-train evolution must reproduce the
dense split-step solver.

At R = 12 with maxdim = 64 the bond dimension never binds, so this isolates
*correctness* from truncation: TCI of the initial state, the potential MPOs, the
Fourier-based kinetic operator, the re-interpolated nonlinearity, the renormalisations
and the Trotter step order all have to be right simultaneously.

The nonlinear case (g != 0) is the one that matters -- it is the only part of the
scheme that cannot be written as a fixed operator, and the only one where the TT and
dense codes share no machinery at all.

The comparison is up to a global phase, which the algorithm does not fix.
"""
import sys

import numpy as np

sys.path.insert(0, "/home/joselado/Documents/programs/tnde")
from tnde import dense, evolve1d, tt

R, XMIN, XMAX, DT = 12, -40.0, 40.0, 0.02
KCUT, BETA, TOL, MAXDIM, NSTEPS = 2**7, 2.0, 1e-12, 64, 40
OMEGA1 = 0.05


def _run(g):
    psi0 = lambda x: ((1 / np.pi) ** 0.25 * np.exp(-(x**2) / 2)).astype(np.complex128)
    V = lambda x: OMEGA1 * x**2
    psi, _ = evolve1d.evolve(psi0, [V], R, XMIN, XMAX, g, DT, NSTEPS,
                             tolerance=TOL, maxdim=MAXDIM, kcut=KCUT,
                             save_every=10, verbose=False)
    d = np.asarray(dense.evolve(psi0, [V], R, XMIN, XMAX, g, DT, NSTEPS,
                                kcut=KCUT, beta=BETA))
    return tt.to_dense(psi), d


def test_matches_dense_linear():
    got, want = _run(0.0)
    ph = np.vdot(want, got) / abs(np.vdot(want, got))
    e = np.max(np.abs(got - ph * want)) / np.max(np.abs(want))
    print(f"  g=0: ||TT - dense|| / ||dense|| = {e:.3e}")
    assert e < 1e-7


def test_matches_dense_nonlinear():
    got, want = _run(5.0)
    ph = np.vdot(want, got) / abs(np.vdot(want, got))
    e = np.max(np.abs(got - ph * want)) / np.max(np.abs(want))
    w_tt = float(dense.width(got, R, XMIN, XMAX))
    w_d = float(dense.width(want, R, XMIN, XMAX))
    print(f"  g=5: ||TT - dense|| / ||dense|| = {e:.3e}")
    print(f"       width  TT = {w_tt:.9f}  dense = {w_d:.9f}")
    assert e < 1e-7
    assert abs(w_tt - w_d) < 1e-7


if __name__ == "__main__":
    for t in (test_matches_dense_linear, test_matches_dense_nonlinear):
        print(f"{t.__name__}:")
        t()
    print("PASS")
