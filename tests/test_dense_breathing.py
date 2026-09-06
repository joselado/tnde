"""Gate for the dense oracle: a Gaussian in a harmonic trap must breathe at the
analytic rate.

For V = w1 x**2 = m w**2 x**2 / 2 with m = 1, the trap frequency is w = sqrt(2 w1).
A minimum-uncertainty Gaussian with <x**2>_0 = a**2, <p**2>_0 = 1/(4a**2) obeys

    <x**2>(t) = <x**2>_0 cos**2(wt) + <p**2>_0 / (m w)**2 * sin**2(wt)

exactly, for any a. The initial state (1/pi)**(1/4) exp(-x**2/2) has a**2 = 1/2.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from tnde import dense


def test_harmonic_breathing():
    R, xmin, xmax, dt = 16, -500.0, 500.0, 0.01
    omega1 = 0.01
    w = np.sqrt(2 * omega1)

    psi0 = lambda x: (1 / np.pi) ** 0.25 * np.exp(-(x**2) / 2)
    V = lambda x: omega1 * x**2

    nsteps = 4000
    steps, snaps = dense.evolve(psi0, [V], R, xmin, xmax, g=0.0, dt=dt,
                                nsteps=nsteps, kcut=2**12, save_every=400)

    print(f"  trap frequency w = {w:.6f}, period {2*np.pi/w:.2f}, "
          f"evolved to T = {(nsteps+1)*dt:.2f}")
    print(f"{'t':>9} {'width (dense)':>15} {'width (exact)':>15} {'rel err':>11}")
    worst = 0.0
    for s, psi in zip(steps, snaps):
        t = (s + 1) * dt                       # Nsteps+1 kinetic applications
        num = float(dense.width(psi, R, xmin, xmax))
        exact = np.sqrt(0.5 * np.cos(w * t) ** 2 + 0.5 / w**2 * np.sin(w * t) ** 2)
        rel = abs(num - exact) / exact
        worst = max(worst, rel)
        print(f"{t:>9.2f} {num:>15.9f} {exact:>15.9f} {rel:>11.2e}")
    print(f"  worst relative error: {worst:.2e}")
    assert worst < 1e-3, f"breathing mismatch: {worst:.2e}"


def test_spatial_convergence():
    """The residual is *spatial*, not Trotter: it is independent of dt and falls as
    O(dx**2). That is the finite-difference dispersion -4M**2/L**2 sin**2(pi k/M) the
    algorithm uses by construction, not a port bug -- so the gate is the convergence
    rate, not a fixed threshold.

    Measured (T = 24, dt = 0.01): dt 0.02 -> 0.0025 leaves the error flat at 8.4e-05,
    while R 14 -> 17 takes it 3.3e-03 -> 2.1e-05, a factor ~4 per doubling.
    """
    xmin, xmax, omega1, dt = -500.0, 500.0, 0.01, 0.01
    w = np.sqrt(2 * omega1)
    psi0 = lambda x: (1 / np.pi) ** 0.25 * np.exp(-(x**2) / 2)
    V = lambda x: omega1 * x**2
    T = 24.0
    n = int(round(T / dt)) - 1
    t = (n + 1) * dt
    exact = np.sqrt(0.5 * np.cos(w * t) ** 2 + 0.5 / w**2 * np.sin(w * t) ** 2)

    errs = {}
    for R in (15, 17):
        psi = dense.evolve(psi0, [V], R, xmin, xmax, g=0.0, dt=dt, nsteps=n, kcut=2**12)
        errs[R] = abs(float(dense.width(psi, R, xmin, xmax)) - exact) / exact
        print(f"  R={R}: rel err {errs[R]:.3e}")
    rate = errs[15] / errs[17]
    print(f"  error ratio over 2 halvings of dx: {rate:.1f} (O(dx**2) predicts ~16)")
    assert errs[17] < 1e-4
    assert rate > 8, f"not converging at the expected order: {rate:.1f}"


if __name__ == "__main__":
    test_harmonic_breathing()
    test_spatial_convergence()
    print("PASS")
