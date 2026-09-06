"""Phase separation of a two-component condensate, by imaginary-time relaxation.

    i d_t psi_a = -1/2 psi_a'' + x**2/2 psi_a + sum_b G[a,b] |psi_b|**2 psi_a,   a = 1, 2

with G = g [[1, r], [r, 1]]. For r > 1 the mixture is immiscible: the two species
expel each other and, with equal populations in a symmetric trap, settle side by
side. Each component keeps its own norm during the relaxation (normalise="each"),
so the populations are fixed rather than transferred to the cheaper species.

Reported: the overlap integral ``int n_1 n_2 dx`` (which drops to ~0 on separation),
each component's centre of mass, and the dense-reference comparison.

Usage:  python examples/binary_mixture_1d.py [R] [r]
"""
import sys
import time

import numpy as np

sys.path.insert(0, "/home/joselado/Documents/programs/tnde")
from tnde import Grid, equations, pde, reference


def main(R=14, r=1.5):
    g = 20.0
    grid = Grid(R, (-12.0, 12.0))
    eq = equations.coupled_gross_pitaevskii(g * np.array([[1.0, r], [r, 1.0]]),
                                            V=lambda x: 0.5 * x**2, imaginary_time=True)
    # a slight left/right bias breaks the mirror symmetry and picks the direction
    psi0 = [lambda x: np.exp(-(x - 0.3) ** 2 / 4), lambda x: np.exp(-(x + 0.3) ** 2 / 4)]
    dt, nsteps = 0.01, 300

    def report(step, t, u):
        x, (n1, n2) = pde.reconstruct(u[0], grid, n=1024)[0], [np.abs(pde.reconstruct(c, grid, n=1024)[1]) ** 2 for c in u]
        dx = x[0][1] - x[0][0]
        ov = np.sum(n1 * n2) * dx
        c1, c2 = np.sum(x[0] * n1) * dx, np.sum(x[0] * n2) * dx
        print(f"    tau = {t:5.2f}  ranks {[c.rank() for c in u]}  overlap {ov:.4f}  "
              f"<x>_1 = {c1:+.3f}  <x>_2 = {c2:+.3f}")

    t0 = time.time()
    u, _ = pde.solve(eq, psi0, grid, dt, nsteps, tolerance=1e-9, maxdim=30, save_every=50,
                     callback=report, verbose=False)
    t_tt = time.time() - t0
    print(f"\n{R = }: {grid.M} points, {nsteps} steps in {t_tt:.1f} s "
          f"({t_tt / nsteps:.2f} s per step, two components)")

    if R <= 16:
        t0 = time.time()
        ud, _ = reference.solve(eq, psi0, grid, dt, nsteps)
        e = max(np.max(np.abs(np.abs(pde.to_dense(a, grid)) - np.abs(b))) / np.max(np.abs(b))
                for a, b in zip(u, ud))
        print(f"  dense reference in {time.time() - t0:.1f} s: max |psi| difference {e:.2e}")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(*([int(args[0])] if args else []), *([float(args[1])] if len(args) > 1 else []))
