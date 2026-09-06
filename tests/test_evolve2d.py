"""Gates for the 2D solver.

The 2D path shares almost nothing with 1D: the state is an *interleaved* train over
x_1 y_1 x_2 y_2 ..., the Fourier transform has to be embedded into a subset of sites,
and the site ordering it produces is not the one it started with. Each of those fails
silently, so all three are checked against a dense 2D FFT.
"""
import sys

import numpy as np
from qutecipy import contract

sys.path.insert(0, "/home/joselado/Documents/programs/tnde")
from tnde import dense, evolve2d
from tnde.fourier import fourier_mpo

R, XMIN, XMAX, YMIN, YMAX = 6, -8.0, 8.0, -8.0, 8.0
DT, M_, KCUT, BETA, TOL, MAXDIM = 0.02, 1.0, 2**3, 2.0, 1e-12, 200

PSI0 = lambda x, y: ((1 / np.pi) ** 0.25
                     * np.exp(-((x - 0.5) ** 2 + (y + 0.7) ** 2) / 2)
                     * np.exp(1j * 1.0 * x)).astype(np.complex128)


def test_interleave_round_trip():
    rng = np.random.default_rng(0)
    ix, iy = rng.integers(0, 1 << R, 500), rng.integers(0, 1 << R, 500)
    jx, jy = evolve2d.batcheval.deinterleave(evolve2d.batcheval.interleave(ix, iy, R), R)
    assert np.array_equal(ix, jx) and np.array_equal(iy, jy)
    print("  interleave/deinterleave round trip over 500 random pairs: ok")


def test_fourier_output_is_transposed():
    """The transform returns an interleaved train over (ky, kx), not (kx, ky).

    Each variable's QFT reverses its own digits in place, and the final whole-train
    reverse swaps which slot belongs to which variable. The original is only correct
    because its momentum operator is symmetric under kx <-> ky; the test function here
    is deliberately asymmetric so the transposition is visible.
    """
    f = lambda x, y: (np.exp(-((x - 0.6) ** 2 + (y + 1.1) ** 2) / 2) * (1 + 0.3 * np.cos(2 * x))
                      + 0.4j * np.exp(-((x + 1.5) ** 2 + (y - 0.4) ** 2)))
    psi = evolve2d.initial_state_2d(f, R, XMIN, XMAX, YMIN, YMAX, 1e-13)
    d = evolve2d.to_dense_2d(psi, R)
    got = evolve2d.to_dense_2d(evolve2d.fourier_transform_2d(psi, R, -1.0, 1e-13), R)
    ref = np.fft.fft2(d) / (1 << R)
    e_t = np.max(np.abs(got - ref.T)) / np.max(np.abs(ref))
    e_n = np.max(np.abs(got - ref)) / np.max(np.abs(ref))
    print(f"  vs fft2 transposed: {e_t:.3e}   vs fft2 as-is: {e_n:.3e}")
    assert e_t < 1e-12 and e_n > 0.1


def test_kinetic_step_matches_dense():
    psi = evolve2d.initial_state_2d(PSI0, R, XMIN, XMAX, YMIN, YMAX, TOL)
    d0 = evolve2d.to_dense_2d(psi, R)
    lap = evolve2d.exp_lap_lowpass_mpo_2d(R, XMIN, XMAX, YMIN, YMAX, DT, M_, KCUT, BETA, TOL)
    qft = fourier_mpo(R, sign=-1.0, tolerance=1e-13)
    iqft = fourier_mpo(R, sign=+1.0, tolerance=1e-13)
    ft = evolve2d.fourier_transform_2d(psi, R, -1.0, TOL, MAXDIM, qft=qft)
    ft = contract(lap, ft, algorithm="naive", tolerance=TOL, maxbonddim=MAXDIM)
    got = evolve2d.to_dense_2d(evolve2d.fourier_transform_2d(ft, R, +1.0, TOL, MAXDIM, qft=iqft), R)
    kfac = np.asarray(dense.kinetic_factor_2d(R, XMIN, XMAX, YMIN, YMAX, DT, M_, KCUT, BETA))
    want = np.fft.ifft2(kfac * np.fft.fft2(d0))
    e = np.max(np.abs(got - want)) / np.max(np.abs(want))
    print(f"  one kinetic step: ||TT - dense|| / ||dense|| = {e:.3e}")
    assert e < 1e-10


def _evolution(g):
    V1 = lambda x, y: 0.05 * (x**2 + y**2)
    V2 = lambda x, y: 2.0 * np.sin(x) ** 2
    psi, _ = evolve2d.evolve(PSI0, [V1, V2], R, XMIN, XMAX, YMIN, YMAX, g, DT, 6,
                             m=M_, tolerance=TOL, maxdim=MAXDIM, kcut=KCUT, beta=BETA,
                             fourier_tol=1e-13, save_every=100, verbose=False)
    d = np.asarray(dense.evolve_2d(PSI0, [V1, V2], R, XMIN, XMAX, YMIN, YMAX, g, DT, 6,
                                   m=M_, kcut=KCUT, beta=BETA))
    got = evolve2d.to_dense_2d(psi, R)
    ph = np.vdot(d, got) / abs(np.vdot(d, got))
    return np.max(np.abs(got - ph * d)) / np.max(np.abs(d))


def test_evolution_matches_dense_linear():
    e = _evolution(0.0)
    print(f"  g=0, 6 steps: ||TT - dense|| / ||dense|| = {e:.3e}")
    assert e < 1e-9


def test_evolution_matches_dense_nonlinear():
    e = _evolution(5.0)
    print(f"  g=5, 6 steps: ||TT - dense|| / ||dense|| = {e:.3e}")
    assert e < 1e-9


def test_moments_match_analytic():
    """The 2D observables of ``observables_2D.jl``, against a case with known answers.

    A displaced Gaussian of width s has <x> = x0, <y> = y0, width s/sqrt(2) in each
    direction and <r**2> = x0**2 + y0**2 + s**2. Both routes are checked: the MPO one
    (which the original uses, and which is limited by a tolerance relative to
    max|x**2|) and direct summation over a reconstructed sample.
    """
    from tnde import observables

    r, tol = 7, 1e-12
    x0, y0, s = 1.5, -0.8, 1.0
    psi0 = lambda x, y: ((1 / (np.pi * s**2)) ** 0.5
                         * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * s**2))
                         ).astype(np.complex128)
    psi = evolve2d.normalise_2d(
        evolve2d.initial_state_2d(psi0, r, -16., 16., -16., 16., tol),
        r, -16., 16., -16., 16.)
    exact = {"x": x0, "y": y0, "width_x": s / np.sqrt(2), "width_y": s / np.sqrt(2),
             "r2": x0**2 + y0**2 + s**2}
    mpo = observables.moments_2d(psi, r, -16., 16., -16., 16., tolerance=tol)
    dense_ = observables.moments_2d_dense(psi, r, -16., 16., -16., 16., prec=7)
    print(f"  {'quantity':>9} {'MPO route':>12} {'direct sum':>12} {'analytic':>12}")
    for k, want in exact.items():
        print(f"  {k:>9} {mpo[k]:>12.6f} {dense_[k]:>12.6f} {want:>12.6f}")
        assert abs(mpo[k] - want) < 1e-5
        assert abs(dense_[k] - want) < 1e-5


if __name__ == "__main__":
    for t in (test_interleave_round_trip, test_fourier_output_is_transposed,
              test_kinetic_step_matches_dense, test_evolution_matches_dense_linear,
              test_evolution_matches_dense_nonlinear, test_moments_match_analytic):
        print(f"{t.__name__}:")
        t()
    print("PASS")
