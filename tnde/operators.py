"""The MPOs the Trotter step is built from.

Everything here follows ``utilities.jl``. Two constructions deserve a note:

* the kinetic propagator is *not* a fixed operator. ``kinetic_mpo`` searches for a
  momentum cutoff, raising it until a Gaussian trial state survives a round trip
  through the low-pass with at least 99% of its norm. At R = 30 this is the most
  expensive object built anywhere in the run, and it happens before step 1.
* the low-pass and the kinetic phase live on a *momentum* grid indexed 0..2**R-1, and
  the resulting diagonal MPO has to be sandwiched between forward and inverse quantics
  Fourier MPOs. Because the QFT inverts site order, three ``reverse`` calls are needed
  and getting any of them wrong fails silently -- see ``tnde.fourier``.
"""
from __future__ import annotations

import numpy as np
from qutecipy import crossinterpolate2, optfirstpivot
from qutecipy.tensortrain.core import reverse
from scipy.special import expit

from tnde import batcheval, tt
from tnde.fourier import fourier_mpo

#: TCI's own default, and what the Julia code falls back on wherever it omits the
#: keyword (``exp_lap_Fourier_MPO_lowpass`` accepts a ``tol`` argument and then never
#: passes it on -- preserved here).
DEFAULT_TOL = 1e-8


#: TCI's global pivot search is the mechanism that finds regions the current
#: interpolation gets wrong. The default of 5 is not enough for a needle-shaped
#: function on a very wide box -- see :func:`peak_pivots` -- so this port raises it.
DEFAULT_NSEARCHGLOBALPIVOT = 20


def _bits(i, R):
    return [(i >> (R - 1 - n)) & 1 for n in range(R)]


def _value(pivot, R):
    return int(np.dot(np.asarray(pivot, dtype=np.int64), 1 << np.arange(R - 1, -1, -1, dtype=np.int64)))


def peak_pivots(f, R, nrandom=5, rng=None, extra=None):
    """Initial pivots: the peaks of ``|f|``, plus their immediate neighbours.

    Two failure modes are being defended against, and both are silent.

    The first is Julia's: the default all-zeros pivot means ``x = xmin``, where a state
    localised near the origin is numerically zero, and TCI aborts with "maxsamplevalue
    is zero". ``quanticscrossinterpolate`` hides this by drawing ``nrandominitpivot``
    random pivots and refining each by greedy coordinate ascent on ``|f|``.

    The second is worse and is *not* fixed by that. A function whose support straddles a
    high-order bit boundary -- a Gaussian centred at ``x = 0`` on ``[-500, 500]`` sits
    exactly on the most significant bit -- can be interpolated on one side only. TCI
    then reports a converged error of 7e-11 while the true error is 1.0, because its
    error estimate is sampled on its own pivots and never looks at the missing half.
    Measured on the paper's initial state: Julia's single centre pivot fails this way in
    1 run out of 6, and adding random pivots does not help (2 out of 6) because greedy
    ascent walks every start to the same peak.

    Including the peak's *neighbours* fixes it, since ``i`` and ``i-1`` lie in opposite
    branches whenever ``i`` is on a bit boundary. This is the same "get pivots in
    non-trivial branches" trick the Julia code applies to the low-pass MPO but not to
    the wave function. With it, 0 runs out of 6 fail.
    """
    rng = rng or np.random.default_rng(0)
    top = (1 << R) - 1
    idxs = set()
    for p in (extra or []):
        idxs.add(_value(p, R))
    for _ in range(nrandom):
        i = _value(optfirstpivot(f, [2] * R, rng.integers(0, 2, size=R).tolist()), R)
        idxs.update((i, max(i - 1, 0), min(i + 1, top)))
    if not idxs:
        idxs.add(0)
    return [_bits(i, R) for i in sorted(idxs)]


def random_init_pivots(f, R, nrandom, rng=None, extra=None):
    """Backwards-compatible alias for :func:`peak_pivots`."""
    return peak_pivots(f, R, nrandom, rng, extra)


def _tci(fn, R, lo, hi, tolerance, pivots=None, maxbonddim=None, cache=True,
         nrandom=0, rng=None, nsearchglobalpivot=DEFAULT_NSEARCHGLOBALPIVOT):
    f = batcheval.grid_function(fn, R, lo, hi, cache=cache)
    if nrandom:
        pivots = peak_pivots(f, R, nrandom, rng, extra=pivots)
    elif pivots is None:
        pivots = [[0] * R]
    kw = {"tolerance": tolerance, "nsearchglobalpivot": nsearchglobalpivot}
    if maxbonddim is not None:
        kw["maxbonddim"] = maxbonddim
    ci, _, _ = crossinterpolate2(np.complex128, f, [2] * R, pivots, **kw)
    return ci


# --------------------------------------------------------------------------
# potential and position operators (real-space, diagonal)
# --------------------------------------------------------------------------

def exp_potential_mpo(V, h, R, xmin, xmax, tolerance=1e-8, nrandom=8):
    """``exp(-i V(x) h)`` as a diagonal MPO (``exp_potential_MPO``).

    Julia passes ``nrandominitpivot=1000`` here; 8 is enough in practice because
    ``|exp(-iVh)| == 1`` everywhere, so every pivot is as good as any other and the
    seeding only needs to break the initial degeneracy. Raise it if a potential ever
    makes the interpolation stall.
    """
    return tt.tt_to_mpo(_tci(lambda x: np.exp(-1j * V(x) * h), R, xmin, xmax, tolerance,
                             nrandom=nrandom))


def pos_mpo(R, xmin, xmax, tolerance=1e-10):
    """``x`` as a diagonal MPO.

    Note the tolerance is *relative to* ``max|x|``, so on the paper's box the absolute
    accuracy is 500x worse than the number passed -- this is why the reference
    ``widths.txt`` is only good to O(0.1). See PORTING_NOTES.md.
    """
    return tt.tt_to_mpo(_tci(lambda x: x.astype(np.complex128), R, xmin, xmax, tolerance))


def pos_squared_mpo(R, xmin, xmax, tolerance=1e-10):
    """``x**2`` as a diagonal MPO. The same relative-tolerance caveat applies, squared."""
    return tt.tt_to_mpo(_tci(lambda x: (x**2).astype(np.complex128), R, xmin, xmax, tolerance))


# --------------------------------------------------------------------------
# momentum-space kinetic operator
# --------------------------------------------------------------------------

def lowpass(k, kcut, kmax, beta):
    """Fermi-Dirac low-pass in DFT index space (``low_pass_MPO_FD``).

    Unity at both *ends* of the index range -- which is where the small physical
    momenta live -- and zero across the middle. ``expit`` keeps it finite for the
    k ~ 2**30 the exponent would otherwise overflow on.
    """
    return 1.0 + expit(-(k - kcut) * beta) - expit(-(k - (kmax - kcut)) * beta)


def exp_lap_lowpass_mpo(R, xmin, xmax, dt, m, kcut, beta, tolerance=DEFAULT_TOL):
    """``lowpass(k) * exp(i dt/(2m) * lap_FD(k))`` as a diagonal MPO in momentum index.

    ``lap_FD(k) = -4 M**2/L**2 sin**2(pi k / M)`` is the finite-difference Laplacian
    eigenvalue, not ``-k**2``; the difference is the O(dx**2) error the whole scheme
    carries and it must be kept for the dense oracle to agree.

    Both trivial branches are given as initial pivots (all-zeros and all-ones), because
    the function is only non-negligible at the two *ends* of the index range and a
    single pivot in the middle would see nothing.
    """
    M = 1 << R
    kmax = M - 1
    L = xmax - xmin

    def f(k):
        phase = np.exp(1j / 2 * (-4.0) * M**2 / (m * L**2) * np.sin(np.pi / M * k) ** 2 * dt)
        return lowpass(k, kcut, kmax, beta) * phase

    # grid step is exactly 1 here, so the grid coordinate *is* the momentum index
    ci = _tci(f, R, 0.0, float(kmax), tolerance, pivots=[[0] * R, [1] * R])
    return tt.tt_to_mpo(ci)


def kinetic_mpo_at(R, xmin, xmax, dt, m, kcut, tolerance, beta=2.0, maxbonddim=100,
                   fourier_tol=1e-12, exp_lap_tol=1e-10, qft=None, iqft=None):
    """The real-space kinetic propagator at a *given* momentum cutoff.

    ``invQFT . (lowpass * exp(lap)) . QFT``. The three ``reverse`` calls undo the QFT's
    site-order inversion: the forward transform leaves the train reversed, so the
    normally ordered diagonal momentum operator must be reversed to act on it, and the
    inverse transform must be reversed to bring the result back to normal order. Get
    any of them wrong and the operator is silently garbage rather than obviously broken.
    """
    from qutecipy import contract

    if qft is None:
        qft = fourier_mpo(R, sign=-1.0, tolerance=fourier_tol)
    if iqft is None:
        iqft = fourier_mpo(R, sign=+1.0, tolerance=fourier_tol)
    lap = exp_lap_lowpass_mpo(R, xmin, xmax, dt, m, kcut, beta, exp_lap_tol)
    op1 = contract(reverse(lap), qft, algorithm="naive",
                   tolerance=tolerance, maxbonddim=maxbonddim)
    return contract(reverse(iqft), op1, algorithm="naive",
                    tolerance=tolerance, maxbonddim=maxbonddim)


def kinetic_mpo(R, xmin, xmax, dt, m, tolerance, beta=2.0, kcut_exp0=8,
                target=0.99, maxbonddim=100, fourier_tol=1e-12, exp_lap_tol=1e-10,
                verbose=True):
    """The full real-space kinetic propagator ``invQFT . lowpass*exp(lap) . QFT``.

    Reproduces ``get_kinetic_lowpass_mpo``: build the operator at ``kcut = 2**kcut_exp``
    starting from ``kcut_exp = 8``, apply it to a Gaussian trial state, and raise the
    cutoff until the state keeps ``target`` of its norm. A cutoff that is too low
    silently eats the wave function's momentum content, so this search is doing real
    work, not tuning.

    The three ``reverse`` calls are the QFT's site-order inversion being undone: the
    forward transform leaves the train reversed, so the (normally ordered) diagonal
    momentum operator has to be reversed to act on it, and the inverse transform has to
    be reversed to bring the result back to normal order.
    """
    from qutecipy import contract

    qft = fourier_mpo(R, sign=-1.0, tolerance=fourier_tol)
    iqft = fourier_mpo(R, sign=+1.0, tolerance=fourier_tol)

    # Gaussian trial state, with the pivot placed at the peak (index 2**(R-1)) --
    # the Julia code uses 1000 random pivots for the same purpose, less reliably.
    p = [0] * R
    p[0] = 1
    trial = tt.from_cores(tt.cores(
        _tci(lambda x: ((1 / np.pi) ** 0.25 * np.exp(-(x**2) / 2)).astype(np.complex128),
             R, xmin, xmax, tolerance, pivots=[p], nrandom=8)))

    n, kcut_exp, op = 0.0, kcut_exp0, None
    while n < target:
        if 2**kcut_exp >= (1 << R) // 2:
            raise RuntimeError(
                f"momentum cutoff reached half the grid (2**{kcut_exp}) without "
                f"retaining {target} of the trial norm; R={R} is too coarse for dt={dt}")
        op = kinetic_mpo_at(R, xmin, xmax, dt, m, 2**kcut_exp, tolerance, beta,
                            maxbonddim, qft=qft, iqft=iqft)
        out = contract(op, trial, algorithm="naive", tolerance=tolerance, maxbonddim=maxbonddim)
        n = tt.fidelity(out, out, R, xmin, xmax)
        if verbose:
            print(f"    kcut = 2**{kcut_exp} = {2**kcut_exp}: trial-state norm retained {n:.6f}"
                  f"  (operator rank {op.rank()})")
        kcut_exp += 1
    return op
