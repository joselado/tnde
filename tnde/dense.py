"""Dense split-step Gross-Pitaevskii solver in JAX.

This is the *oracle* for the tensor-train port, not the port itself: it stores the
full 2**R-point wave function, so it is usable up to R ~ 20 and hopeless at the
flagship R = 30 (10**9 points). Its job is to pin down the conventions -- grid
spacing, the finite-difference dispersion, the Fourier normalisation, the exact
order of the Trotter half-steps -- before the tensor-train machinery can hide a
convention error, and to provide a reference the TT solver must reproduce at small R.

Every convention here is copied from the Julia original (``utilities.jl``,
``GP_1D.jl``) including its internal inconsistencies, which are preserved
deliberately -- see PORTING_NOTES.md:

* the spatial grid uses ``includeendpoint=true`` spacing ``(xmax-xmin)/(2**R - 1)``,
* but ``normalise`` and ``fidelity`` use ``dx = (xmax-xmin)/2**R``,
* and the kinetic phase uses the finite-difference dispersion
  ``-4M**2/L**2 sin**2(pi k/M)``, which corresponds to yet a third spacing ``L/M``.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from tnde import config  # noqa: F401  (enables x64 on import)


# --------------------------------------------------------------------------
# grid
# --------------------------------------------------------------------------

def xgrid(R: int, xmin: float, xmax: float) -> jnp.ndarray:
    """The 2**R real-space grid points, ``includeendpoint=True`` convention.

    ``x[0] == xmin`` and ``x[-1] == xmax`` exactly, i.e. spacing ``(xmax-xmin)/(2**R-1)``.
    """
    M = 1 << R
    return xmin + jnp.arange(M, dtype=jnp.float64) * (xmax - xmin) / (M - 1)


def dx_norm(R: int, xmin: float, xmax: float) -> float:
    """The ``dx`` used by ``normalise``/``fidelity_ITensor`` -- ``L/2**R``, *not* the
    grid spacing. Preserved from the original; see the module docstring."""
    return (xmax - xmin) / (1 << R)


# --------------------------------------------------------------------------
# operators
# --------------------------------------------------------------------------

def lowpass(k: jnp.ndarray, kcut: float, kmax: int, beta: float) -> jnp.ndarray:
    """Fermi-Dirac low-pass in DFT index space (``low_pass_MPO_FD``).

    Unity near ``k = 0`` and ``k = kmax`` -- the two ends of the DFT index range,
    which are the *small* physical momenta -- and zero across the middle. Written
    with ``sigmoid`` rather than Julia's ``1/(exp(z)+1)`` so that ``k`` up to 2**30
    does not overflow.
    """
    s = jax.nn.sigmoid
    return 1.0 + s(-(k - kcut) * beta) - s(-(k - (kmax - kcut)) * beta)


def kinetic_factor(R: int, xmin: float, xmax: float, dt: float, m: float,
                   kcut: float, beta: float) -> jnp.ndarray:
    """``lowpass(k) * exp(i dt/(2m) * lap_FD(k))`` -- the momentum-space kinetic
    propagator, exactly ``lap_Fourier_lowpass`` from ``utilities.jl``.

    The dispersion is the *finite-difference* Laplacian eigenvalue
    ``-4 M**2/L**2 sin**2(pi k/M)``, not ``-k**2``.
    """
    M = 1 << R
    L = xmax - xmin
    k = jnp.arange(M, dtype=jnp.float64)
    phase = jnp.exp(1j / 2 * (-4.0) * M**2 / (m * L**2) * jnp.sin(jnp.pi / M * k) ** 2 * dt)
    return lowpass(k, kcut, M - 1, beta) * phase


# --------------------------------------------------------------------------
# steps
# --------------------------------------------------------------------------

def normalise(psi: jnp.ndarray, R: int, xmin: float, xmax: float) -> jnp.ndarray:
    """Divide by ``n = fidelity**(1/4) = sqrt(sum |psi|**2 dx)``.

    The Julia version spreads ``1/n`` over the R MPS cores as ``n**(-1/R)`` each;
    the product is the same scalar, so dense and TT agree.
    """
    n = jnp.sqrt(jnp.sum(jnp.abs(psi) ** 2) * dx_norm(R, xmin, xmax))
    return psi / n


def kinetic_step(psi: jnp.ndarray, kfac: jnp.ndarray) -> jnp.ndarray:
    """One full kinetic step: forward DFT, multiply, inverse DFT.

    ``quanticsfouriermpo(normalize=true)`` is the *unitary* DFT (each core carries a
    ``1/sqrt(2)``, so ``1/sqrt(M)`` overall) and its inverse multiplies by
    ``sqrt(M)``; the two factors cancel, so the plain ``fft``/``ifft`` pair here is
    the same operator.
    """
    return jnp.fft.ifft(kfac * jnp.fft.fft(psi))


def nonlinear_step(psi: jnp.ndarray, g: float, h: float) -> jnp.ndarray:
    """``exp(-i g |psi|**2 h) psi`` -- the Gross-Pitaevskii nonlinearity (``apply_f_tt``)."""
    return jnp.exp(-1j * g * jnp.abs(psi) ** 2 * h) * psi


def potential_step(psi: jnp.ndarray, vphase: jnp.ndarray) -> jnp.ndarray:
    """``exp(-i V(x) h) psi``, with the phase precomputed by :func:`potential_phase`."""
    return vphase * psi


def potential_phase(V, x: jnp.ndarray, h: float) -> jnp.ndarray:
    return jnp.exp(-1j * V(x) * h)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

@partial(jax.jit, static_argnums=(1, 6))
def _evolve(psi, R, kfac, vhalf, vfull, params, nsteps):
    """The Trotter loop, matching ``GP_Trotter_1D``'s exact step order:

        (V/2)(g/2)  [ K N (g) (V) ]**Nsteps  K N (g/2) (V/2)

    where ``N`` is the renormalisation the original performs after every kinetic
    step. Total evolved time is ``(Nsteps + 1) * dt`` -- there are Nsteps+1 kinetic
    applications, not Nsteps.
    """
    xmin, xmax, g, dt = params

    psi = vhalf * psi
    psi = jax.lax.cond(g != 0.0, lambda p: nonlinear_step(p, g, dt / 2), lambda p: p, psi)
    psi = normalise(psi, R, xmin, xmax)

    def body(_, p):
        p = kinetic_step(p, kfac)
        p = normalise(p, R, xmin, xmax)
        p = jax.lax.cond(g != 0.0, lambda q: nonlinear_step(q, g, dt), lambda q: q, p)
        return vfull * p

    psi = jax.lax.fori_loop(0, nsteps, body, psi)

    psi = kinetic_step(psi, kfac)
    psi = normalise(psi, R, xmin, xmax)
    psi = jax.lax.cond(g != 0.0, lambda p: nonlinear_step(p, g, dt / 2), lambda p: p, psi)
    psi = vhalf * psi
    return normalise(psi, R, xmin, xmax)


def evolve(psi0, potentials, R, xmin, xmax, g, dt, nsteps, m=1.0,
           kcut=2**8, beta=2.0, save_every=None):
    """Evolve ``psi0`` under ``sum(potentials)`` for ``nsteps`` Trotter steps.

    ``psi0`` and each entry of ``potentials`` are callables on the real-space grid.
    Returns the final state, or -- when ``save_every`` is given -- a
    ``(nsnapshots, 2**R)`` array of snapshots taken every ``save_every`` steps,
    with the same step indexing the Julia code saves under (1, then every
    ``save_every``-th).
    """
    x = xgrid(R, xmin, xmax)
    psi = normalise(jnp.asarray(psi0(x), dtype=jnp.complex128), R, xmin, xmax)
    kfac = kinetic_factor(R, xmin, xmax, dt, m, kcut, beta)

    def total_V(xx):
        out = jnp.zeros_like(xx)
        for V in potentials:
            out = out + V(xx)
        return out

    vhalf = potential_phase(total_V, x, dt / 2)
    vfull = potential_phase(total_V, x, dt)
    params = (xmin, xmax, g, dt)

    if save_every is None:
        return _evolve(psi, R, kfac, vhalf, vfull, params, nsteps)

    # Snapshots: re-enter the loop in chunks. The half-steps at each end mean a
    # chunked run is *not* identical to one long run, so each snapshot is produced
    # by its own complete (half-step, loop, half-step) evolution from psi0 -- which
    # is what the Julia code's saved files represent anyway.
    steps = [1] + list(range(save_every, nsteps + 1, save_every))
    return steps, jnp.stack([_evolve(psi, R, kfac, vhalf, vfull, params, s) for s in steps])


# --------------------------------------------------------------------------
# observables
# --------------------------------------------------------------------------

def width(psi: jnp.ndarray, R: int, xmin: float, xmax: float) -> float:
    """``sqrt(<x**2> - <x>**2)`` with the original's ``dx = L/2**R`` weighting."""
    x = xgrid(R, xmin, xmax)
    dx = dx_norm(R, xmin, xmax)
    rho = jnp.abs(psi) ** 2 * dx
    x1 = jnp.sum(rho * x)
    x2 = jnp.sum(rho * x**2)
    return jnp.sqrt(x2 - x1**2)


# --------------------------------------------------------------------------
# 2D oracle
# --------------------------------------------------------------------------

def xygrid(R: int, xmin, xmax, ymin, ymax):
    """The 2**R x 2**R grid. ``GP_2D.jl`` builds its grid *without* ``includeendpoint``,
    so the spacing is ``L/2**R`` and the last point is one step short of the upper
    bound -- unlike the 1D code. Preserved."""
    M = 1 << R
    x = xmin + jnp.arange(M, dtype=jnp.float64) * (xmax - xmin) / M
    y = ymin + jnp.arange(M, dtype=jnp.float64) * (ymax - ymin) / M
    return jnp.meshgrid(x, y, indexing="ij")


def kinetic_factor_2d(R, xmin, xmax, ymin, ymax, dt, m, kcut, beta):
    """``lowpass(kx) lowpass(ky) exp(i dt/(2m) (lap_FD(kx) + lap_FD(ky)))``.

    Note the original uses ``L = xmax - xmin`` for *both* directions
    (``exp_lap_Fourier_MPO_lowpass_2D`` never reads the y bounds), which is only
    correct for a square box. Preserved.
    """
    M = 1 << R
    L = xmax - xmin
    k = jnp.arange(M, dtype=jnp.float64)
    kx, ky = jnp.meshgrid(k, k, indexing="ij")
    phase = jnp.exp(1j / 2 * (-4.0) * M**2 / (m * L**2)
                    * (jnp.sin(jnp.pi / M * kx) ** 2 + jnp.sin(jnp.pi / M * ky) ** 2) * dt)
    return lowpass(kx, kcut, M - 1, beta) * lowpass(ky, kcut, M - 1, beta) * phase


def normalise_2d(psi, R, xmin, xmax, ymin, ymax):
    dxdy = (xmax - xmin) / (1 << R) * (ymax - ymin) / (1 << R)
    return psi / jnp.sqrt(jnp.sum(jnp.abs(psi) ** 2) * dxdy)


def evolve_2d(psi0, potentials, R, xmin, xmax, ymin, ymax, g, dt, nsteps,
              m=1.0, kcut=2**8, beta=2.0):
    """Dense split-step 2D evolution, in ``GP_Trotter_MPS_2D``'s step order:

        (V/2)(g/2)  [ K N (V)(g) ]**nsteps  K N (V/2)(g/2)

    Note the potential and nonlinear steps are swapped relative to the 1D driver --
    they are both diagonal in real space and commute, so this is cosmetic, but it is
    what the original does.
    """
    X, Y = xygrid(R, xmin, xmax, ymin, ymax)
    psi = normalise_2d(jnp.asarray(psi0(X, Y), dtype=jnp.complex128), R, xmin, xmax, ymin, ymax)
    kfac = kinetic_factor_2d(R, xmin, xmax, ymin, ymax, dt, m, kcut, beta)
    V = sum(V(X, Y) for V in potentials)
    vhalf, vfull = jnp.exp(-1j * V * dt / 2), jnp.exp(-1j * V * dt)

    def kinetic(p):
        return jnp.fft.ifft2(kfac * jnp.fft.fft2(p))

    psi = vhalf * psi
    if g != 0:
        psi = nonlinear_step(psi, g, dt / 2)
    for _ in range(nsteps):
        psi = normalise_2d(kinetic(psi), R, xmin, xmax, ymin, ymax)
        psi = vfull * psi
        if g != 0:
            psi = nonlinear_step(psi, g, dt)
    psi = normalise_2d(kinetic(psi), R, xmin, xmax, ymin, ymax)
    psi = vhalf * psi
    if g != 0:
        psi = nonlinear_step(psi, g, dt / 2)
    return psi
