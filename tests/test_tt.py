"""Gate for the tensor-train layer: the reference MPS from the paper must round-trip
through it -- correct values, correct norm, correct elementwise-product MPO."""
import sys

import numpy as np
from qutecipy import DiscretizedGrid, CachedFunction, crossinterpolate2, contract

sys.path.insert(0, "/home/joselado/Documents/programs/tnde")
from tnde import tt

REF = "/home/joselado/Documents/programs/tnde/refdata/ref_mps_1D.npz"
R, XMIN, XMAX = 30, -500.0, 500.0


def ref_state(step):
    z = np.load(REF)
    return tt.from_cores([z[f"step{step}_core{i}"] for i in range(int(z[f"step{step}_n"]))])


def test_evaluate_matches_analytic_gaussian():
    """step1 is the TCI'd, normalised Gaussian. Evaluating it must reproduce
    (1/pi)**(1/4) exp(-x**2/2) wherever that is above the TCI noise floor."""
    psi = ref_state(1)
    dx = (XMAX - XMIN) / (2**R - 1)          # includeendpoint spacing
    i0 = 2 ** (R - 1)                        # x ~ 0
    offs = np.array([0, 1, 10**6, 5 * 10**6, 10**7])
    vals = tt.evaluate(psi, i0 + offs)
    x = XMIN + (i0 + offs) * dx
    exact = (1 / np.pi) ** 0.25 * np.exp(-(x**2) / 2)
    for xx, v, e in zip(x, vals, exact):
        note = "" if e > 1e-12 else "   <- below TCI noise floor, excluded from the check"
        print(f"  x={xx:>12.6f}  |mps|={abs(v):.9e}  exact={e:.9e}  "
              f"ratio={abs(v)/e:.9f}{note}")
    # Only where the exact value is above the TCI noise floor: tolerance is relative
    # to max|f| ~ 0.75, so an absolute floor of ~1e-14 is expected and the last point
    # (exact = 1e-19) is pure noise, not a mismatch.
    ok = exact > 1e-12
    assert np.allclose(np.abs(vals[ok]), exact[ok], rtol=2e-6)


def test_reference_is_normalised():
    """Julia renormalised before saving, so fidelity(psi,psi) should already be ~1."""
    for step in (1, 10, 50, 100):
        f = tt.fidelity(ref_state(step), ref_state(step), R, XMIN, XMAX)
        print(f"  step {step:>3}: fidelity = {f:.12f}  (1 - f = {1-f:+.2e})")
        # Julia renormalises after the kinetic step but then applies the nonlinear and
        # potential MPOs before saving; those are unitary only up to the TCI/truncation
        # error, so a drift of ~1e-8 by the time the state is written out is expected.
        assert abs(f - 1.0) < 1e-6


def test_normalise_is_idempotent():
    psi = ref_state(10)
    scaled = tt.from_cores([c * 3.7 for c in tt.cores(psi)])
    n = tt.normalise(scaled, R, XMIN, XMAX)
    print(f"  fidelity after renormalising a 3.7x-scaled state: "
          f"{tt.fidelity(n, n, R, XMIN, XMAX):.12f}")
    assert abs(tt.fidelity(n, n, R, XMIN, XMAX) - 1.0) < 1e-12
    # and it really is the same state up to that scale
    assert abs(abs(tt.inner(n, psi)) / np.sqrt(abs(tt.inner(n, n) * tt.inner(psi, psi))) - 1) < 1e-12


def test_tt_to_mpo_is_elementwise_multiplication():
    """Contracting tt_to_mpo(f) with g must give the pointwise product f*g.
    Checked densely at R=10, where the full 2**R vector is available."""
    r = 10
    grid = DiscretizedGrid.from_resolutions(["x"], [r], lower_bound=(-4.0,),
                                            upper_bound=(4.0,), includeendpoint=True)

    def tci_of(fn):
        q = lambda b: complex(fn(grid.quantics_to_origcoord(b)[0]))
        cf = CachedFunction(np.complex128, q, [2] * r)
        ci, _, _ = crossinterpolate2(np.complex128, cf, [2] * r, [[0] * r], tolerance=1e-12)
        return ci

    f = lambda x: np.exp(-(x**2) / 2)
    g = lambda x: np.cos(3 * x) + 2.0
    ttf, ttg = tci_of(f), tci_of(g)
    prod = contract(tt.tt_to_mpo(ttf), tt.from_cores(tt.cores(ttg)),
                    algorithm="naive", tolerance=1e-14)

    x = np.array([grid.grididx_to_origcoord(i)[0] for i in range(2**r)])
    got, want = tt.to_dense(prod), f(x) * g(x)
    err = np.max(np.abs(got - want)) / np.max(np.abs(want))
    print(f"  max relative error of the elementwise product: {err:.3e}")
    assert err < 1e-11


if __name__ == "__main__":
    for t in (test_evaluate_matches_analytic_gaussian, test_reference_is_normalised,
              test_normalise_is_idempotent, test_tt_to_mpo_is_elementwise_multiplication):
        print(f"{t.__name__}:")
        t()
    print("PASS")
