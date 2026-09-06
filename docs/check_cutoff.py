"""Reproduce the momentum-cutoff measurements in docs/physics.md.

Everything here uses the dense oracle, so it runs in a few seconds at moderate R.
The DFT-index-to-momentum map ``k_phys = 2 pi k / L`` depends only on the box length,
not on R, so a cutoff calibrated at R=16 is the one the R=30 run selects.

Usage:  python docs/check_cutoff.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tnde import dense


def dispersion_limit(R=16, L=1000.0):
    """lambda_FD(k) -> -(2 pi k / L)**2 at small k."""
    M = 1 << R
    k = np.arange(8)
    lam = -4.0 * M**2 / L**2 * np.sin(np.pi / M * k) ** 2
    print("continuum limit of the finite-difference dispersion:")
    print(f"  max rel. deviation from -(2 pi k/L)**2 over k=1..7: "
          f"{np.max(np.abs(lam[1:] / (-(2*np.pi*k[1:]/L)**2) - 1)):.2e}")


def cutoff_search(R=16, xmin=-500.0, xmax=500.0, dt=0.01, m=1.0, beta=2.0):
    """Which kcut the search in operators.kinetic_mpo accepts, and why."""
    M, L = 1 << R, xmax - xmin
    x = np.asarray(dense.xgrid(R, xmin, xmax))
    psi = (1 / np.pi) ** 0.25 * np.exp(-(x**2) / 2)      # the fixed trial state
    dx = L / M
    print(f"\ncutoff search (criterion: trial-state fidelity >= 0.99), L={L}:")
    for e in range(8, 12):
        kf = np.asarray(dense.kinetic_factor(R, xmin, xmax, dt, m, 2**e, beta))
        out = np.fft.ifft(kf * np.fft.fft(psi))
        fid = (np.sum(np.abs(out) ** 2) * dx) ** 2
        print(f"  kcut=2**{e:<2} k_phys_cut={2*np.pi*2**e/L:8.3f}  fidelity={fid:.6f}"
              f"  {'accepted' if fid >= 0.99 else 'rejected'}")
        if fid >= 0.99:
            break


def cutoff_cost_1d(R=18, xmin=-500.0, xmax=500.0, dt=0.01, g=5.0, m=1.0, nsteps=100):
    """What the accepted cutoff costs on the paper's own 1D configuration."""
    M, L = 1 << R, xmax - xmin
    x = np.asarray(dense.xgrid(R, xmin, xmax))
    dx = L / M
    V = 0.01 * x**2 + 5.0 * np.sin(10 * x) ** 2
    vhalf, vfull = np.exp(-1j * V * dt / 2), np.exp(-1j * V * dt)

    def run(kcut):
        kf = np.asarray(dense.kinetic_factor(R, xmin, xmax, dt, m, kcut, 2.0))
        p = ((1 / np.pi) ** 0.25 * np.exp(-(x**2) / 2)).astype(np.complex128)
        p /= np.sqrt(np.sum(np.abs(p) ** 2) * dx)
        p = np.exp(-1j * g * np.abs(vhalf * p) ** 2 * dt / 2) * (vhalf * p)
        p /= np.sqrt(np.sum(np.abs(p) ** 2) * dx)
        loss = []
        for _ in range(nsteps):                      # mid-Trotter, as snapshots are
            p = np.fft.ifft(kf * np.fft.fft(p))
            n = np.sum(np.abs(p) ** 2) * dx
            loss.append(1 - n)
            p /= np.sqrt(n)
            p = vfull * (np.exp(-1j * g * np.abs(p) ** 2 * dt) * p)
        return p, np.asarray(loss)

    def width(p):
        w = np.abs(p) ** 2 / np.sum(np.abs(p) ** 2)
        return np.sqrt(np.sum(w * x**2) - np.sum(w * x) ** 2)

    print(f"\npaper's 1D configuration, {nsteps} steps, dense R={R}:")
    print(f"  {'kcut':>8} {'k_phys_cut':>11} {'max loss/step':>15} {'cumulative':>12} {'width':>9}")
    out = {}
    for e in (9, 12, 14):
        p, loss = run(2**e)
        out[e] = p
        print(f"  {'2**%d' % e:>8} {2*np.pi*2**e/L:>11.2f} {loss.max():>15.3e} "
              f"{loss.sum():>12.3e} {width(p):>9.4f}")
    a, b = out[9], out[14]
    ov = abs(np.vdot(a, b)) / np.sqrt(np.vdot(a, a).real * np.vdot(b, b).real)
    print(f"  1 - overlap(2**9, 2**14) = {1-ov:.3e}")


if __name__ == "__main__":
    dispersion_limit()
    cutoff_search()
    cutoff_cost_1d()
