"""The 1D Gross-Pitaevskii Trotter evolution (``GP_Trotter_1D`` from ``GP_1D.jl``).

The scheme is a second-order mixed-spectral splitting: the kinetic part is applied in
momentum space through the quantics Fourier MPO, the potential and the ``g|psi|**2``
nonlinearity in real space. The step order, including the renormalisation after every
kinetic application and the half-steps at each end, follows the original exactly:

    (V/2)(g/2)  [ K N (g)(V) ]**nsteps  K N (g/2)(V/2)

so ``nsteps`` loop iterations mean ``nsteps + 1`` kinetic applications and a total
evolved time of ``(nsteps + 1) * dt``.

The nonlinearity cannot be applied as a fixed operator -- it depends on the state --
so each step re-interpolates ``exp(-i g |psi|**2 dt) psi`` from scratch. That is the
dominant cost per step and the reason the batched, memoising evaluator in
``gptci.batcheval`` exists.
"""
from __future__ import annotations

import os
import pickle
import time

import numpy as np
from qutecipy import contract, crossinterpolate2

from gptci import batcheval, tt
from gptci.operators import (DEFAULT_NSEARCHGLOBALPIVOT, exp_potential_mpo,
                             kinetic_mpo, kinetic_mpo_at, peak_pivots)


def initial_state(psi0, R, xmin, xmax, tolerance):
    """TCI the initial wave function, with the pivot forced to the grid centre.

    ``GP_1D.jl`` sets ``p1[1] = 2`` for exactly this reason: the default all-ones pivot
    sits at ``x = xmin``, where a state localised near the origin is numerically zero,
    and TCI started there can fail to find the structure at all.

    That single pivot is not enough, though. It lands exactly on the most significant
    bit, and TCI then interpolates only the ``x > 0`` half of the Gaussian in 1 run out
    of 6 -- reporting a converged 7e-11 error while the true error is 1.0. Seeding
    *both* straddling indices ``2**(R-1)`` and ``2**(R-1) - 1``, which lie in opposite
    MSB branches, removes it. See :func:`gptci.operators.peak_pivots`.
    """
    f = batcheval.grid_function(psi0, R, xmin, xmax)
    centre = 1 << (R - 1)
    straddle = [[(i >> (R - 1 - n)) & 1 for n in range(R)] for i in (centre, centre - 1)]
    ci, _, _ = crossinterpolate2(np.complex128, f, [2] * R, straddle,
                                 tolerance=tolerance,
                                 nsearchglobalpivot=DEFAULT_NSEARCHGLOBALPIVOT)
    return tt.from_cores(tt.cores(ci))


def apply_nonlinearity(psi, R, xmin, xmax, g, h, tolerance, maxdim,
                       nrandom=5, rng=None):
    """Re-interpolate ``exp(-i g |psi|**2 h) psi`` (Julia's ``apply_f_tt``).

    The pivot seeding is not optional here. The state is localised near the middle of
    a very wide box, so the default all-zeros pivot sits where the wave function is
    numerically zero and TCI aborts outright ("maxsamplevalue is zero"). Julia avoids
    this via ``quanticscrossinterpolate``'s ``nrandominitpivot=5``; the same seeding is
    reproduced in :func:`gptci.operators.random_init_pivots`.
    """
    fn = lambda wf, x: np.exp(-1j * g * np.abs(wf) ** 2 * h) * wf
    f = batcheval.tt_function(psi, fn, R, xmin, xmax, D=maxdim)
    ci, _, _ = crossinterpolate2(np.complex128, f, [2] * R,
                                 peak_pivots(f, R, nrandom, rng),
                                 tolerance=tolerance, maxbonddim=maxdim,
                                 nsearchglobalpivot=DEFAULT_NSEARCHGLOBALPIVOT)
    return tt.from_cores(tt.cores(ci))


def _cached_kinetic(R, xmin, xmax, dt, m, tolerance, beta, cachedir, kcut=None, **kw):
    """The kcut search is the most expensive object in a run and depends only on
    ``(R, xmin, xmax, dt, m, beta, tolerance)``, so it is worth keeping on disk.

    Passing an explicit ``kcut`` skips the search entirely -- useful for reusing a
    cutoff already known to be adequate, and for comparing against the dense solver at
    a *matched* cutoff.
    """
    def build():
        if kcut is not None:
            return kinetic_mpo_at(R, xmin, xmax, dt, m, kcut, tolerance, beta=beta)
        return kinetic_mpo(R, xmin, xmax, dt, m, tolerance, beta=beta, **kw)

    if cachedir is None:
        return build()
    os.makedirs(cachedir, exist_ok=True)
    key = f"kin_R{R}_x{xmin}_{xmax}_dt{dt}_m{m}_b{beta}_tol{tolerance}_k{kcut}.pkl"
    path = os.path.join(cachedir, key)
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return tt.from_cores(pickle.load(fh))
    op = build()
    with open(path, "wb") as fh:
        pickle.dump([np.asarray(op.sitetensor(n)) for n in range(len(op))], fh)
    return op


def evolve(psi0, potentials, R, xmin, xmax, g, dt, nsteps, m=1.0, tolerance=1e-10,
           maxdim=14, beta=2.0, exp_pot_tol=1e-8, save_every=10, cachedir=None,
           kcut=None, verbose=True):
    """Run the evolution and return ``(final_state, snapshots)``.

    ``snapshots`` is a dict keyed the way the original names its files: ``1`` is the
    initial normalised state (saved *before* the opening half-steps) and every
    ``save_every``-th key is the state after that loop iteration, i.e. after the
    kinetic, nonlinear and full-``dt`` potential applications -- mid-Trotter, not
    symmetrised. That is what the reference ``.jld2`` files contain.
    """
    log = print if verbose else (lambda *a, **k: None)
    t0 = time.time()

    psi = tt.normalise(initial_state(psi0, R, xmin, xmax, tolerance), R, xmin, xmax)
    log(f"  initial state: rank {psi.rank()}  ({time.time()-t0:.1f}s)")

    half = [exp_potential_mpo(V, dt / 2, R, xmin, xmax, exp_pot_tol) for V in potentials]
    full = [exp_potential_mpo(V, dt, R, xmin, xmax, exp_pot_tol) for V in potentials]
    log(f"  potential MPOs: ranks {[o.rank() for o in full]}  ({time.time()-t0:.1f}s)")

    log("  building kinetic operator (searching for the momentum cutoff):")
    kin = _cached_kinetic(R, xmin, xmax, dt, m, tolerance, beta, cachedir,
                          kcut=kcut, verbose=verbose)
    log(f"  kinetic operator: rank {kin.rank()}  ({time.time()-t0:.1f}s)")

    snapshots = {1: psi}

    def apply_mpo(state, mpo):
        return contract(mpo, state, algorithm="naive", tolerance=tolerance, maxbonddim=maxdim)

    for o in half:
        psi = apply_mpo(psi, o)
    if g != 0:
        psi = apply_nonlinearity(psi, R, xmin, xmax, g, dt / 2, tolerance, maxdim)
    psi = tt.normalise(psi, R, xmin, xmax)

    for j in range(1, nsteps + 1):
        psi = apply_mpo(psi, kin)
        psi = tt.normalise(psi, R, xmin, xmax)
        if g != 0:
            psi = apply_nonlinearity(psi, R, xmin, xmax, g, dt, tolerance, maxdim)
        for o in full:
            psi = apply_mpo(psi, o)
        if j % save_every == 0:
            snapshots[j] = psi
            log(f"  step {j:>5}/{nsteps}  rank {psi.rank():>3}  "
                f"norm {tt.fidelity(psi, psi, R, xmin, xmax):.9f}  ({time.time()-t0:.1f}s)")

    psi = apply_mpo(psi, kin)
    psi = tt.normalise(psi, R, xmin, xmax)
    if g != 0:
        psi = apply_nonlinearity(psi, R, xmin, xmax, g, dt / 2, tolerance, maxdim)
    for o in half:
        psi = apply_mpo(psi, o)
    psi = tt.normalise(psi, R, xmin, xmax)

    log(f"  done in {time.time()-t0:.1f}s")
    return psi, snapshots
