"""A droplet shrinking under the 2D Allen-Cahn equation -- a case the train does
*not* compress well, kept as an honest example of the cost of sharp curved fronts.

    d_t u = eps lap u + u - u**3

A disk of u = +1 in a sea of u = -1 shrinks by mean-curvature flow, v = -eps / r,
so its radius obeys r(t)**2 = r0**2 - 2 eps t: an analytic check. But a circular
interface of width sqrt(2 eps) has no low-rank quantics representation -- it is not
separable and not smooth -- so the bond dimension is set by the tolerance (rank ~50
at 1e-4, ~90 at 1e-8 on a 256 x 256 grid) rather than by the physics, and each step
costs seconds where a smooth field costs a fraction of one. Compare
``examples/advection_diffusion_2d.py``.

Usage:  python examples/allen_cahn_droplet_2d.py [R]
"""
import sys
import time

import numpy as np

sys.path.insert(0, "/home/joselado/Documents/programs/tnde")
from tnde import Grid, equations, pde, reference


def radius(u, grid, n=256):
    """Radius of the u > 0 region from its area, on an n x n sample."""
    (x, y), v = pde.reconstruct(u, grid, n=n)
    area = np.mean(v.real > 0) * grid.lengths[0] * grid.lengths[1]
    return np.sqrt(area / np.pi)


def main(R=7):
    L, eps, r0 = 2 * np.pi, 0.02, 1.5
    grid = Grid(R, [(0.0, L), (0.0, L)])
    w = np.sqrt(2 * eps)
    u0 = lambda x, y: np.tanh((r0 - np.sqrt((x - np.pi) ** 2 + (y - np.pi) ** 2)) / w)
    dt, nsteps = 0.1, 100

    def report(step, t, u):
        r = radius(u, grid)
        print(f"    t = {t:5.1f}  rank {u.rank():>3}  radius {r:.4f}  "
              f"curvature-flow prediction {np.sqrt(r0**2 - 2 * eps * t):.4f}")

    t0 = time.time()
    u, _ = pde.solve(equations.allen_cahn(eps=eps), u0, grid, dt, nsteps, tolerance=1e-5,
                     maxdim=60, save_every=20, callback=report, verbose=False)
    t_tt = time.time() - t0
    print(f"\n{R = }: {grid.M}x{grid.M} grid, {nsteps} steps in {t_tt:.1f} s "
          f"({t_tt / nsteps:.2f} s per step)")
    t0 = time.time()
    ud, _ = reference.solve(equations.allen_cahn(eps=eps), u0, grid, dt, nsteps)
    e = np.max(np.abs(pde.to_dense(u, grid) - ud)) / np.max(np.abs(ud))
    print(f"  dense reference in {time.time() - t0:.1f} s: max difference {e:.2e} "
          f"(the truncation at tolerance 1e-5)")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 7)
