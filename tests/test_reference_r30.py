"""The strongest gate available: reproduce the paper's own R = 30 run.

Parameters are exactly ``GP_1D.jl``'s -- 2**30 grid points on [-500, 500], a Gaussian
in a harmonic trap plus a fast sinusoidal lattice, g = 5, dt = 0.01, maxdim = 14 --
and the result is compared against the MPS tensors committed to the original
repository (converted from JLD2, see refdata/).

Slow: ~60 s for 10 steps. The full 100-step comparison lives in
examples/reproduce_paper_1d.py, which reaches 1 - overlap = 2.2e-10 at step 100.
"""
import sys

import numpy as np

sys.path.insert(0, "/home/joselado/Documents/programs/tnde")
from gptci import evolve1d, observables, tt

R, XMIN, XMAX = 30, -500.0, 500.0
DT, TOL, MAXDIM, G = 0.01, 1e-10, 14, 5.0
REF = "/home/joselado/Documents/programs/tnde/refdata/ref_mps_1D.npz"


def test_reproduces_reference():
    psi0 = lambda x: ((1 / np.pi) ** 0.25 * np.exp(-(x**2) / 2)).astype(np.complex128)
    pots = [lambda x: 0.01 * x**2, lambda x: 5.0 * np.sin(10 * x) ** 2]
    _, snaps = evolve1d.evolve(psi0, pots, R, XMIN, XMAX, G, DT, 10,
                               tolerance=TOL, maxdim=MAXDIM, save_every=10, verbose=False)
    z = np.load(REF)
    for s in sorted(snaps):
        a = snaps[s]
        b = tt.from_cores([z[f"step{s}_core{i}"] for i in range(int(z[f"step{s}_n"]))])
        ov = abs(tt.inner(a, b)) / np.sqrt(abs(tt.inner(a, a) * tt.inner(b, b)))
        print(f"  step {s:>3}: rank {a.rank()} (ref {b.rank()})  1 - overlap = {1-ov:.3e}")
        assert a.rank() == b.rank()
        assert 1 - ov < 1e-9


def test_initial_width_is_analytic():
    """The initial normalised Gaussian has width exactly sqrt(1/2).

    The reference ``widths.txt`` says 0.8117 because its x**2 MPO is built at a
    tolerance relative to max|x**2| = 2.5e5; ``width_dense`` avoids that entirely.
    """
    psi0 = lambda x: ((1 / np.pi) ** 0.25 * np.exp(-(x**2) / 2)).astype(np.complex128)
    psi = tt.normalise(evolve1d.initial_state(psi0, R, XMIN, XMAX, TOL), R, XMIN, XMAX)
    w = observables.width_dense(psi, R, XMIN, XMAX, halfwidth=50.0)
    print(f"  width = {w:.9f}   analytic sqrt(1/2) = {np.sqrt(0.5):.9f}   "
          f"(reference widths.txt says 0.8117)")
    assert abs(w - np.sqrt(0.5)) < 1e-6


if __name__ == "__main__":
    for t in (test_initial_width_is_analytic, test_reproduces_reference):
        print(f"{t.__name__}:")
        t()
    print("PASS")
