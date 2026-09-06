"""The 2D Gross-Pitaevskii Trotter evolution (``GP_Trotter_MPS_2D`` from ``GP_2D.jl``).

The state is an interleaved quantics train over ``x_1 y_1 x_2 y_2 ... x_R y_R``, so the
kinetic step cannot use a single R-site Fourier MPO the way 1D does. Each variable is
transformed separately by embedding the R-site QFT MPO into the 2R-site chain with
identity cores at the other variable's positions.

Two conventions differ from the 1D code and must not be unified:

* the 2D grid is built *without* ``includeendpoint``, so its spacing is ``L/2**R`` --
  which happens to agree with the ``dx`` that ``normalise_2D`` uses, where the 1D pair
  disagree;
* the momentum operator uses ``L = xmax - xmin`` for *both* directions, so it is only
  correct on a square box.
"""
from __future__ import annotations

import time

import numpy as np
from qutecipy import contract, crossinterpolate2
from qutecipy.tensortrain.core import TensorTrain, reverse

from tnde import batcheval, fit, tt
from tnde.tt import embed_mpo  # noqa: F401  (re-exported; lived here originally)
from tnde.fourier import fourier_mpo
from tnde.operators import DEFAULT_NSEARCHGLOBALPIVOT, peak_pivots, lowpass


# --------------------------------------------------------------------------
# 2D Fourier transform
# --------------------------------------------------------------------------

def fourier_transform_2d(psi, R, sign=-1.0, tolerance=1e-10, maxbonddim=None,
                         fourier_tol=1e-12, qft=None, method="naive", nsweeps=2):
    """Transform both variables of an interleaved 2D train.

    Reproduces ``Fourier_transform_2D``: apply the QFT to the x sites, then to the y
    sites, then reverse the whole train.

    The site ordering that comes out is the subtle part. Each QFT leaves *its own*
    variable's digits in reversed significance order in place (ITensor hides this by
    relabelling indices rather than moving tensors), and the final whole-train reverse
    both undoes that and swaps which slot belongs to which variable. The net result is
    an interleaved train in normal MSB-first order over ``(ky, kx)`` -- the two
    variables exchanged. The original gets away with it because the 2D momentum
    operator is symmetric under ``kx <-> ky``; an asymmetric one would be silently
    transposed.
    """
    n = len(tt.cores(psi))
    if n != 2 * R:
        raise ValueError(f"expected an interleaved 2R={2*R}-site train, got {n}")
    if qft is None:
        qft = fourier_mpo(R, sign=sign, tolerance=fourier_tol)
    cores = [np.asarray(qft.sitetensor(i)) for i in range(R)]

    out = psi
    for offset in (0, 1):
        mpo = embed_mpo(cores, range(offset, 2 * R, 2), 2 * R)
        out = _apply(mpo, out, method, tolerance, maxbonddim, nsweeps)
    return reverse(out)


def _apply(mpo, state, method, tolerance, maxbonddim, nsweeps=2):
    """Apply an MPO, by variational fit or by exact product plus global SVD.

    Which one to use is not uniform, and the split matters:

    * the **momentum-space kinetic MPO** needs the fit. It has rank ~65 against a state
      at maxdim 50, so the exact intermediate carries bond dimensions near 3000:
      measured at R=12, ``naive`` takes 171 s against the fit's 1.4 s -- 123x -- and the
      two agree to 2.2e-08.
    * the **Fourier MPO** must *not* use it. It is highly non-local, so two sweeps from a
      truncated-state guess converge poorly: on the R=6 gate the fit gives 2.8e-02 where
      naive gives 7.5e-13. It is also cheap either way (1.6 s at R=20), so ``naive`` costs
      nothing here. The original makes the same split -- ``Quantics.fouriertransform``
      goes through ITensor's accurate ``apply``, while only ``apply_MPO_IT`` uses
      ``method="fit"``.
    * **1D** stays on ``naive`` throughout: its MPOs have rank <= 17, so the exact
      product is small and slightly more accurate.
    """
    if method == "fit" and maxbonddim is not None:
        return fit.apply_mpo(mpo, state, maxbonddim, nsweeps=nsweeps, tolerance=tolerance)
    return contract(mpo, state, algorithm="naive", tolerance=tolerance,
                    maxbonddim=maxbonddim)


# --------------------------------------------------------------------------
# operators
# --------------------------------------------------------------------------

def exp_potential_mpo_2d(V, h, R, xmin, xmax, ymin, ymax, tolerance=1e-8, nrandom=8,
                         rng=None):
    """``exp(-i V(x,y) h)`` as a diagonal MPO on the interleaved train."""
    f = batcheval.grid_function_2d(lambda x, y: np.exp(-1j * V(x, y) * h),
                                   R, xmin, xmax, ymin, ymax)
    ci, _, _ = crossinterpolate2(np.complex128, f, [2] * (2 * R),
                                 peak_pivots(f, 2 * R, nrandom, rng),
                                 tolerance=tolerance,
                                 nsearchglobalpivot=DEFAULT_NSEARCHGLOBALPIVOT)
    return tt.tt_to_mpo(ci)


def exp_lap_lowpass_mpo_2d(R, xmin, xmax, ymin, ymax, dt, m, kcut, beta,
                           tolerance=1e-8, rng=None):
    """The 2D momentum-space kinetic factor as a diagonal MPO over interleaved
    ``(kx, ky)``.

    All four corner branches are seeded as pivots: the factor is only non-negligible
    where *both* momenta are small, which on the DFT index range means near
    ``(0,0)``, ``(0,kmax)``, ``(kmax,0)`` and ``(kmax,kmax)``. This is what
    ``exp_lap_Fourier_MPO_lowpass_2D``'s ``initialpivots`` are for.
    """
    M = 1 << R
    kmax = M - 1
    L = xmax - xmin

    def f(kx, ky):
        phase = np.exp(1j / 2 * (-4.0) * M**2 / (m * L**2)
                       * (np.sin(np.pi / M * kx) ** 2 + np.sin(np.pi / M * ky) ** 2) * dt)
        return lowpass(kx, kcut, kmax, beta) * lowpass(ky, kcut, kmax, beta) * phase

    # grid step is exactly 1, so the coordinate is the momentum index
    fn = batcheval.grid_function_2d(f, R, 0.0, float(M), 0.0, float(M))
    corners = [batcheval.interleave(a, b, R) for a in (0, kmax) for b in (0, kmax)]
    pivots = [[(int(i) >> (2 * R - 1 - n)) & 1 for n in range(2 * R)] for i in corners]
    ci, _, _ = crossinterpolate2(np.complex128, fn, [2] * (2 * R), pivots,
                                 tolerance=tolerance,
                                 nsearchglobalpivot=DEFAULT_NSEARCHGLOBALPIVOT)
    return tt.tt_to_mpo(ci)


# --------------------------------------------------------------------------
# state helpers
# --------------------------------------------------------------------------

def fidelity_2d(a, b, R, xmin, xmax, ymin, ymax) -> float:
    """``|<a|b> dx dy|**2`` (``fidelity_ITensor_2D``)."""
    dxdy = (xmax - xmin) / (1 << R) * (ymax - ymin) / (1 << R)
    return float(abs(tt.inner(a, b) * dxdy) ** 2)


def normalise_2d(psi, R, xmin, xmax, ymin, ymax) -> TensorTrain:
    """Spread ``1/n`` over all ``2R`` cores (``normalise_2D``)."""
    cs = tt.cores(psi)
    n = np.sqrt(np.sqrt(fidelity_2d(psi, psi, R, xmin, xmax, ymin, ymax)))
    n_R = n ** (1.0 / len(cs))
    return tt.from_cores([c / n_R for c in cs])


def initial_state_2d(psi0, R, xmin, xmax, ymin, ymax, tolerance, nrandom=8, rng=None):
    f = batcheval.grid_function_2d(psi0, R, xmin, xmax, ymin, ymax)
    centre = batcheval.interleave(1 << (R - 1), 1 << (R - 1), R)
    seeds = [int(centre), max(int(centre) - 1, 0), int(centre) + 1]
    extra = [[(i >> (2 * R - 1 - n)) & 1 for n in range(2 * R)] for i in seeds]
    ci, _, _ = crossinterpolate2(np.complex128, f, [2] * (2 * R),
                                 peak_pivots(f, 2 * R, nrandom, rng, extra=extra),
                                 tolerance=tolerance,
                                 nsearchglobalpivot=DEFAULT_NSEARCHGLOBALPIVOT)
    return tt.from_cores(tt.cores(ci))


def apply_nonlinearity_2d(psi, R, xmin, xmax, ymin, ymax, g, h, tolerance, maxdim,
                          nrandom=5, rng=None):
    """Re-interpolate ``exp(-i g |psi|**2 h) psi`` (``apply_f_tt_2D``)."""
    fn = lambda wf, x, y: np.exp(-1j * g * np.abs(wf) ** 2 * h) * wf
    f = batcheval.tt_function_2d(psi, fn, R, xmin, xmax, ymin, ymax, D=maxdim)
    ci, _, _ = crossinterpolate2(np.complex128, f, [2] * (2 * R),
                                 peak_pivots(f, 2 * R, nrandom, rng),
                                 tolerance=tolerance, maxbonddim=maxdim,
                                 nsearchglobalpivot=DEFAULT_NSEARCHGLOBALPIVOT)
    return tt.from_cores(tt.cores(ci))


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def evolve(psi0, potentials, R, xmin, xmax, ymin, ymax, g, dt, nsteps, m=1.0,
           tolerance=1e-8, maxdim=50, kcut=2**8, beta=2.0, fourier_tol=1e-8,
           save_every=10, method="fit", nsweeps=2, verbose=True):
    """Run the 2D evolution and return ``(final_state, snapshots)``.

    Step order follows ``GP_Trotter_MPS_2D``:

        (V/2)(g/2)  [ K N (V)(g) ]**nsteps  K N (V/2)(g/2)

    -- note the potential and nonlinear steps are in the opposite order to the 1D
    driver. They commute (both diagonal in real space), so this is cosmetic.
    ``kcut`` is fixed here, not searched, again as in the original.
    """
    log = print if verbose else (lambda *a, **k: None)
    t0 = time.time()

    psi = normalise_2d(initial_state_2d(psi0, R, xmin, xmax, ymin, ymax, tolerance),
                       R, xmin, xmax, ymin, ymax)
    log(f"  initial state: rank {psi.rank()}  ({time.time()-t0:.1f}s)")

    half = [exp_potential_mpo_2d(V, dt / 2, R, xmin, xmax, ymin, ymax, tolerance)
            for V in potentials]
    full = [exp_potential_mpo_2d(V, dt, R, xmin, xmax, ymin, ymax, tolerance)
            for V in potentials]
    log(f"  potential MPOs: ranks {[o.rank() for o in full]}  ({time.time()-t0:.1f}s)")

    lap = exp_lap_lowpass_mpo_2d(R, xmin, xmax, ymin, ymax, dt, m, kcut, beta, tolerance)
    qft = fourier_mpo(R, sign=-1.0, tolerance=fourier_tol)
    iqft = fourier_mpo(R, sign=+1.0, tolerance=fourier_tol)
    log(f"  momentum MPO: rank {lap.rank()}  ({time.time()-t0:.1f}s)")

    def apply_mpo(state, mpo):
        return _apply(mpo, state, method, tolerance, maxdim, nsweeps)

    def kinetic(state):
        # The Fourier transforms deliberately do NOT take `method`: they must stay on
        # `naive` even when the operator applications use the fit. See `_apply`.
        ft = fourier_transform_2d(state, R, -1.0, tolerance, maxdim, qft=qft)
        ft = apply_mpo(ft, lap)
        return fourier_transform_2d(ft, R, +1.0, tolerance, maxdim, qft=iqft)

    snapshots = {1: psi}
    for o in half:
        psi = apply_mpo(psi, o)
    if g != 0:
        psi = apply_nonlinearity_2d(psi, R, xmin, xmax, ymin, ymax, g, dt / 2,
                                    tolerance, maxdim)

    for j in range(1, nsteps + 1):
        psi = normalise_2d(kinetic(psi), R, xmin, xmax, ymin, ymax)
        for o in full:
            psi = apply_mpo(psi, o)
        if g != 0:
            psi = apply_nonlinearity_2d(psi, R, xmin, xmax, ymin, ymax, g, dt,
                                        tolerance, maxdim)
        if j % save_every == 0:
            snapshots[j] = psi
            log(f"  step {j:>4}/{nsteps}  rank {psi.rank():>3}  ({time.time()-t0:.1f}s)")

    psi = normalise_2d(kinetic(psi), R, xmin, xmax, ymin, ymax)
    for o in half:
        psi = apply_mpo(psi, o)
    if g != 0:
        psi = apply_nonlinearity_2d(psi, R, xmin, xmax, ymin, ymax, g, dt / 2,
                                    tolerance, maxdim)
    log(f"  done in {time.time()-t0:.1f}s")
    return psi, snapshots


def to_dense_2d(psi, R):
    """The full 2**R x 2**R array. Small R only -- for the tests."""
    M = 1 << R
    ix, iy = np.meshgrid(np.arange(M), np.arange(M), indexing="ij")
    return tt.evaluate(psi, batcheval.interleave(ix.ravel(), iy.ravel(), R)).reshape(M, M)
