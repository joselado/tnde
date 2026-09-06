"""Ground state of a 1D Bose gas by imaginary-time evolution on a 2**R-point grid.

    i d_t psi = -1/2 psi'' + x**2/2 psi + g |psi|**2 psi

relaxed in imaginary time from a Gaussian guess. With g = 50 the cloud is deep in
the Thomas-Fermi regime, so the converged density can be compared with the
Thomas-Fermi profile n(x) = (mu - x**2/2)/g, mu = (3g/2)**(2/3)/2, and the chemical
potential with its Thomas-Fermi value. The dense reference solver (numpy FFT on the
full grid) runs the identical scheme for comparison.

Usage:  python examples/ground_state_gp.py [R] [g]
"""
import sys
import time

import numpy as np

sys.path.insert(0, "/home/joselado/Documents/programs/tnde")
from tnde import Grid, equations, pde, reference


def main(R=16, g=50.0):
    grid = Grid(R, (-16.0, 16.0))
    eq = equations.gross_pitaevskii(V=lambda x: 0.5 * x**2, g=g, imaginary_time=True)
    psi0 = lambda x: np.exp(-x**2 / 8)
    dt, nsteps = 0.01, 400

    t0 = time.time()
    psi, snaps = pde.solve(eq, psi0, grid, dt, nsteps, tolerance=1e-10, maxdim=30,
                           save_every=100)
    t_tt = time.time() - t0

    x, = grid.mesh()
    rho = np.abs(pde.to_dense(psi, grid)) ** 2
    mu_tf = (1.5 * g) ** (2 / 3) / 2
    n_tf = np.clip(mu_tf - x**2 / 2, 0, None) / g

    # chemical potential mu = <H> with the mean field counted once, from the full grid
    k, = grid.kmesh()
    psi_d = pde.to_dense(psi, grid)
    ekin = 0.5 * np.sum(k**2 * np.abs(np.fft.fft(psi_d)) ** 2) / grid.M * grid.dvol
    mu = ekin + np.sum((0.5 * x**2 + g * rho) * rho) * grid.dvol

    print(f"\n{R = }: {grid.M} grid points, {nsteps} steps of dt = {dt} in {t_tt:.1f} s "
          f"({t_tt / nsteps:.3f} s per step), final rank {psi.rank()}")
    print(f"  chemical potential   mu = {mu:.5f}   Thomas-Fermi {mu_tf:.5f}")
    print(f"  peak density         {rho.max():.5f}   Thomas-Fermi {n_tf.max():.5f}")
    print(f"  cloud radius (rms)   {np.sqrt(np.sum(x**2 * rho) * grid.dvol):.5f}   "
          f"Thomas-Fermi {np.sqrt(2 * mu_tf / 5):.5f}")

    if R <= 18:
        t0 = time.time()
        psi_ref, _ = reference.solve(eq, psi0, grid, dt, nsteps)
        e = np.max(np.abs(np.abs(psi_d) - np.abs(psi_ref))) / np.max(np.abs(psi_ref))
        print(f"  dense reference in {time.time() - t0:.1f} s: max |psi| difference {e:.2e}")


if __name__ == "__main__":
    main(*(float(a) if i else int(a) for i, a in enumerate(sys.argv[1:])))
