"""A smooth field on a 2D grid too large to store: advection-diffusion of a Gaussian
on 2**R x 2**R points, checked against the exact solution.

    d_t u = -(c . grad) u + D lap u

A Gaussian stays a Gaussian: it drifts with c and its variances grow as
s**2 + 2 D t, so every snapshot can be compared with a closed form. Being smooth
and nearly separable, the field is low-rank at every step, and the cost is set by
that rank, not by the 2**(2R) grid points: R = 12 is 16.8 million points, R = 15 is
10**9, and neither exists in memory. (Sharp curved fronts are the opposite case --
see examples/allen_cahn_droplet_2d.py.)

Usage:  python examples/advection_diffusion_2d.py [R]
"""
import sys
import time

import numpy as np

sys.path.insert(0, "/home/joselado/Documents/programs/tnde")
from tnde import Grid, equations, pde


def main(R=12):
    D, c = 0.05, (1.0, -0.6)
    sx, sy = 0.7, 1.1
    grid = Grid(R, [(-20.0, 20.0), (-20.0, 20.0)])
    u0 = lambda x, y: np.exp(-x**2 / (2 * sx**2) - y**2 / (2 * sy**2))

    def exact(x, y, t):
        vx, vy = sx**2 + 2 * D * t, sy**2 + 2 * D * t
        return (sx * sy / np.sqrt(vx * vy)
                * np.exp(-(x - c[0] * t) ** 2 / (2 * vx) - (y - c[1] * t) ** 2 / (2 * vy)))

    def report(step, t, u):
        (x, y), v = pde.reconstruct(u, grid, n=256, window=[(-6, 10), (-10, 6)])
        X, Y = np.meshgrid(x, y, indexing="ij")
        e = np.max(np.abs(v - exact(X, Y, t))) / np.max(np.abs(exact(X, Y, t)))
        print(f"    t = {t:4.1f}  rank {u.rank():>3}  max rel. error vs exact {e:.2e}")

    dt, nsteps = 0.1, 50
    t0 = time.time()
    u, _ = pde.solve(equations.advection_diffusion(c, D=D), u0, grid, dt, nsteps,
                     tolerance=1e-8, maxdim=40, save_every=10, callback=report, verbose=False)
    t_tt = time.time() - t0
    print(f"\n{R = }: {grid.M}x{grid.M} = {grid.M**2:.3g} points, {nsteps} steps in {t_tt:.1f} s "
          f"({t_tt / nsteps:.2f} s per step)")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12)
