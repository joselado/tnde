"""Reproduce the paper's 1D run and check it against the committed reference MPS.

Parameters are exactly those in ``GP_1D.jl``: R = 30 (2**30 grid points on
[-500, 500]), a Gaussian in a harmonic trap plus a fast sinusoidal lattice, g = 5,
dt = 0.01, maxdim = 14.

Usage:  python examples/reproduce_paper_1d.py [nsteps]
"""
import sys
import time

import numpy as np

sys.path.insert(0, "/home/joselado/Documents/programs/tnde")
from gptci import evolve1d, observables, tt

R, XMIN, XMAX = 30, -500.0, 500.0
DT, M, TOL, MAXDIM, G = 0.01, 1.0, 1e-10, 14, 5.0
OMEGA1, OMEGA2, A2 = 0.01, 10, 5.0

PSI0 = lambda x: ((1 / np.pi) ** 0.25 * np.exp(-(x**2) / 2)).astype(np.complex128)
POTENTIALS = [lambda x: OMEGA1 * x**2, lambda x: A2 * np.sin(OMEGA2 * x) ** 2]
REF = "/home/joselado/Documents/programs/tnde/refdata/ref_mps_1D.npz"


def main(nsteps=100):
    t0 = time.time()
    psi, snaps = evolve1d.evolve(PSI0, POTENTIALS, R, XMIN, XMAX, G, DT, nsteps,
                                 m=M, tolerance=TOL, maxdim=MAXDIM, save_every=10,
                                 cachedir="/home/joselado/Documents/programs/tnde/opcache")
    print(f"\n{nsteps} steps in {time.time()-t0:.1f}s "
          f"({(time.time()-t0)/max(nsteps,1):.2f}s per step)")

    z = np.load(REF)
    def ref(s):
        return tt.from_cores([z[f"step{s}_core{i}"] for i in range(int(z[f"step{s}_n"]))])

    print(f"\n{'step':>6} {'rank':>6} {'ref rank':>9} {'1 - |<mine|ref>|':>19} "
          f"{'max |psi| rel. diff':>21} {'width':>10}")
    for s in sorted(snaps):
        if f"step{s}_n" not in z:
            continue
        a, b = snaps[s], ref(s)
        ov = abs(tt.inner(a, b)) / np.sqrt(abs(tt.inner(a, a) * tt.inner(b, b)))
        _, va = observables.reconstruct_window(a, R, XMIN, XMAX, 0.0, 20.0, prec=12)
        _, vb = observables.reconstruct_window(b, R, XMIN, XMAX, 0.0, 20.0, prec=12)
        rel = np.max(np.abs(np.abs(va) - np.abs(vb))) / np.max(np.abs(vb))
        w = observables.width_dense(a, R, XMIN, XMAX, halfwidth=50.0)
        print(f"{s:>6} {a.rank():>6} {b.rank():>9} {1-ov:>19.3e} {rel:>21.3e} {w:>10.6f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100)
