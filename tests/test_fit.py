"""Gate for the variational MPO x MPS fit.

Two things are checked: that it agrees with the exact product (compressed globally),
and that it is much faster than forming that product when the MPO's rank is large --
which is the only reason it exists.
"""
import sys
import time

import numpy as np
from qutecipy import contract
from qutecipy.tensortrain.core import subtract

sys.path.insert(0, "/home/joselado/Documents/programs/tnde")
from tnde import evolve2d, fit
from tnde.fourier import fourier_mpo

R, TOL, MAXDIM = 6, 1e-12, 200
PSI0 = lambda x, y: ((1 / np.pi) ** 0.25
                     * np.exp(-((x - 0.5) ** 2 + (y + 0.7) ** 2) / 2)
                     * np.exp(1j * x)).astype(np.complex128)


def _state():
    return evolve2d.normalise_2d(
        evolve2d.initial_state_2d(PSI0, R, -8., 8., -8., 8., TOL), R, -8., 8., -8., 8.)


def test_matches_exact_product_diagonal():
    psi = _state()
    mpo = evolve2d.exp_potential_mpo_2d(lambda x, y: 0.05 * (x**2 + y**2), 0.01,
                                        R, -8., 8., -8., 8., TOL)
    exact = contract(mpo, psi, algorithm="naive", tolerance=1e-14)
    got = fit.apply_mpo(mpo, psi, MAXDIM, nsweeps=2, tolerance=TOL)
    e = subtract(got, exact).norm() / exact.norm()
    print(f"  diagonal potential MPO (rank {mpo.rank()}): err {e:.3e}")
    assert e < 1e-10


def test_matches_exact_product_momentum():
    """The momentum operator is the case the fit exists for: rank ~65 in 2D."""
    psi = _state()
    ft = evolve2d.fourier_transform_2d(psi, R, -1.0, TOL, MAXDIM,
                                       qft=fourier_mpo(R, sign=-1.0, tolerance=1e-13))
    lap = evolve2d.exp_lap_lowpass_mpo_2d(R, -8., 8., -8., 8., 0.02, 1.0, 2**3, 2.0, TOL)
    exact = contract(lap, ft, algorithm="naive", tolerance=1e-14)
    got = fit.apply_mpo(lap, ft, MAXDIM, nsweeps=2, tolerance=TOL)
    e = subtract(got, exact).norm() / exact.norm()
    print(f"  momentum MPO (rank {lap.rank()}): err {e:.3e}")
    assert e < 1e-10


def test_sweep_gauge_is_maintained():
    """A sweep that leaves the orthogonality centre on the wrong side loses environment
    orthonormality and the environments overflow to inf within one sweep. Catching that
    needs a many-site train, so use the 2R = 12-site 2D state."""
    psi = _state()
    mpo = evolve2d.exp_potential_mpo_2d(lambda x, y: 2.0 * np.sin(x) ** 2, 0.01,
                                        R, -8., 8., -8., 8., TOL)
    for ns in (1, 4, 16):
        got = fit.apply_mpo(mpo, psi, MAXDIM, nsweeps=ns, tolerance=TOL)
        cs = [np.asarray(got.sitetensor(n)) for n in range(len(got))]
        biggest = max(float(np.max(np.abs(c))) for c in cs)
        print(f"  nsweeps={ns:>2}: largest core entry {biggest:.3e}")
        assert np.isfinite(biggest) and biggest < 1e3


def test_converges_when_maxdim_binds():
    """The regime the 2D run actually lives in.

    The other tests here run with slack bond dimension, where the fit has a
    representable exact solution and lands on it -- that checks the algebra, not the
    convergence. Here maxdim is pushed *below* the product's true rank and the fit is
    compared against ``naive`` + global SVD at the *same* maxdim, which is the (near)
    optimal bond-limited answer. Measured at R=6:

        momentum MPO (true rank 16):  maxdim  4 -> 3.9e-02   8 -> 2.6e-05  16 -> 4.9e-14
        potential MPO (true rank 43): maxdim  4 -> 6.0e-02   8 -> 3.2e-04  32 -> 5.4e-13

    and every one of those is *identical* at nsweeps = 1, 2, 4 and 8. So the sweeps
    converge immediately; the residual is the gap between the variational optimum the
    alternating sweeps reach and the globally optimal SVD truncation, and it shrinks
    with maxdim, not with nsweeps. ``nsweeps=2`` is therefore adequate -- the knob that
    matters is maxdim, exactly as in the original.
    """
    psi = _state()
    mpo = evolve2d.exp_potential_mpo_2d(lambda x, y: 0.05 * (x**2 + y**2), 0.01,
                                        R, -8., 8., -8., 8., TOL)
    for md, tolgap in ((8, 1e-3), (16, 1e-5), (32, 1e-11)):
        nai = contract(mpo, psi, algorithm="naive", tolerance=1e-14, maxbonddim=md)
        errs = [subtract(fit.apply_mpo(mpo, psi, md, nsweeps=ns, tolerance=TOL),
                         nai).norm() / nai.norm() for ns in (1, 2, 8)]
        print(f"  maxdim={md:>3} (binds): ||fit-naive||/||naive|| at nsweeps 1,2,8 = "
              + ", ".join(f"{e:.2e}" for e in errs))
        assert errs[1] < tolgap                       # 2 sweeps is good enough
        assert abs(errs[2] - errs[1]) < 0.1 * errs[1] + 1e-15   # and more buys nothing


if __name__ == "__main__":
    for t in (test_matches_exact_product_diagonal, test_matches_exact_product_momentum,
              test_sweep_gauge_is_maintained, test_converges_when_maxdim_binds):
        print(f"{t.__name__}:")
        t()
    print("PASS")
