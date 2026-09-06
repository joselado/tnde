"""The paper's 2D run (``GP_2D.jl``): R = 20, i.e. a 2**20 x 2**20 = 10**12 point grid.

A moving Gaussian in a shallow harmonic trap plus a four-fold quasi-periodic lattice
(two axis-aligned sine modulations and two rotated by 45 degrees).

Usage:  python examples/reproduce_paper_2d.py [nsteps]

There is no committed 2D reference data in the original repository, so correctness
rests on tests/test_evolve2d.py, which pins the 2D path against a dense FFT at small R.
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from tnde import evolve2d, observables

R, XMIN, XMAX, YMIN, YMAX = 20, -100.0, 100.0, -100.0, 100.0
TOL, DT, G, M, MAXDIM, KCUT = 1e-8, 0.01, 5.0, 1.0, 50, 2**8

PSI0 = lambda x, y: ((1 / np.pi) ** 0.25 * np.exp(-(x**2 + y**2) / 2) * np.exp(5j * x))
A2, W2 = 10.0, 1.0
POTENTIALS = [
    lambda x, y: 0.001 * (x**2 + y**2),
    lambda x, y: A2 * np.sin(W2 * x) ** 2,
    lambda x, y: A2 * np.sin(W2 * y) ** 2,
    lambda x, y: A2 * np.sin(W2 / np.sqrt(2) * (x + y)) ** 2,
    lambda x, y: A2 * np.sin(W2 / np.sqrt(2) * (y - x)) ** 2,
]


def main(nsteps=20):
    t0 = time.time()
    psi, snaps = evolve2d.evolve(PSI0, POTENTIALS, R, XMIN, XMAX, YMIN, YMAX, G, DT,
                                 nsteps, m=M, tolerance=TOL, maxdim=MAXDIM, kcut=KCUT,
                                 save_every=5)
    print(f"\n{nsteps} steps in {time.time()-t0:.1f}s "
          f"({(time.time()-t0)/max(nsteps,1):.1f}s per step)")
    for s in sorted(snaps):
        _, _, rho = observables.density_2d(snaps[s], R, XMIN, XMAX, YMIN, YMAX,
                                           prec=7, window=(0.0, 0.0, 12.0))
        print(f"  step {s:>4}: rank {snaps[s].rank():>3}  peak density {rho.max():.6f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
