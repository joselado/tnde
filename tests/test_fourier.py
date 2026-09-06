"""Gate for the quantics Fourier MPO.

The trap this catches is the QFT's intrinsic site-order inversion: without the
``reverse`` the operator is silently wrong (73% error), not obviously broken.
The test function is deliberately *asymmetric and complex* -- for a real even
function ``fft`` and ``ifft*M`` coincide, so a symmetric test cannot tell the two
signs apart and would pass with the sign wrong.
"""
import sys

import numpy as np
from qutecipy import DiscretizedGrid, CachedFunction, crossinterpolate2, contract
from qutecipy.tensortrain.core import reverse

sys.path.insert(0, "/home/joselado/Documents/programs/tnde")
from gptci import tt
from gptci.fourier import fourier_mpo

R = 10
M = 1 << R


def _state():
    grid = DiscretizedGrid.from_resolutions(["x"], [R], lower_bound=(-4.0,),
                                            upper_bound=(4.0,), includeendpoint=True)
    f = lambda x: np.exp(-((x - 0.7) ** 2) / 2) * (1 + 0.3 * np.cos(5 * x)) \
                  + 0.4j * np.exp(-((x + 1.3) ** 2))
    q = lambda b: complex(f(grid.quantics_to_origcoord(b)[0]))
    ci, _, _ = crossinterpolate2(np.complex128, CachedFunction(np.complex128, q, [2] * R),
                                 [2] * R, [[0] * R], tolerance=1e-13)
    return tt.from_cores(tt.cores(ci))


def _apply(mpo, psi):
    return reverse(contract(mpo, psi, algorithm="naive", tolerance=1e-14))


def test_signs_and_normalisation():
    psi = _state()
    d = tt.to_dense(psi)
    fwd_ref, inv_ref = np.fft.fft(d) / np.sqrt(M), np.fft.ifft(d) * np.sqrt(M)
    sep = np.max(np.abs(fwd_ref - inv_ref)) / np.max(np.abs(fwd_ref))
    print(f"  the two references differ by {sep:.3e}, so the signs are distinguishable")
    assert sep > 0.1

    for sign, ref, other, name in ((-1.0, fwd_ref, inv_ref, "forward"),
                                   (+1.0, inv_ref, fwd_ref, "inverse")):
        got = tt.to_dense(_apply(fourier_mpo(R, sign=sign), psi))
        e = np.max(np.abs(got - ref)) / np.max(np.abs(ref))
        e_other = np.max(np.abs(got - other)) / np.max(np.abs(other))
        print(f"  sign={sign:+.0f} ({name:>7}): vs correct {e:.3e}, vs wrong-sign {e_other:.3e}")
        assert e < 1e-13 and e_other > 0.1


def test_reversal_is_required():
    """Without reverse() the operator is wrong by O(1) -- the one silent failure mode."""
    psi = _state()
    ref = np.fft.fft(tt.to_dense(psi)) / np.sqrt(M)
    raw = contract(fourier_mpo(R, sign=-1.0), psi, algorithm="naive", tolerance=1e-14)
    e = np.max(np.abs(tt.to_dense(raw) - ref)) / np.max(np.abs(ref))
    print(f"  un-reversed output error: {e:.3e} (must be O(1), not small)")
    assert e > 0.1


def test_round_trip():
    psi = _state()
    d = tt.to_dense(psi)
    back = _apply(fourier_mpo(R, sign=+1.0), _apply(fourier_mpo(R, sign=-1.0), psi))
    e = np.max(np.abs(tt.to_dense(back) - d)) / np.max(np.abs(d))
    print(f"  inverse(forward(psi)) vs psi: {e:.3e}")
    assert e < 1e-12


if __name__ == "__main__":
    for t in (test_signs_and_normalisation, test_reversal_is_required, test_round_trip):
        print(f"{t.__name__}:")
        t()
    print("PASS")
