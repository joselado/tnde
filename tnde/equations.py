"""Ready-made :class:`~tnde.pde.Equation` objects for common PDEs.

Each function returns an ``Equation`` in the form ``d_t u = l(k) u + p(x) u + N(u)``
of :mod:`tnde.pde`. Where the local ODE has a closed-form solution it is supplied as
an exact ``flow`` -- one evaluation per point and no time-stepping error -- otherwise
the right-hand side is integrated by RK4. Every symbol is written for any number of
dimensions: ``k2(*k)`` is ``|k|**2``.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit

from tnde.pde import Equation


def k2(*k):
    return sum(np.asarray(a, dtype=np.float64) ** 2 for a in k)


def fermi_lowpass(kcut: float, beta: float | None = None):
    """A separable Fermi-Dirac window, ``prod_v 1 / (1 + exp(beta (|k_v| - kcut)))``.

    Oscillatory propagators ``exp(-i h k**2 / 2m)`` have no low-rank quantics form once
    the phase at the grid's largest momentum exceeds a few radians per index -- the
    quantics ranks of the Schrodinger kinetic step grow with ``R`` without a window.
    Physically the window discards momentum content above ``kcut``; it is *not*
    norm-preserving, so the norm after a step is a diagnostic only if the window is
    known to lie above everything the dynamics excites.

    ``beta`` defaults to ``20 / kcut``, a transition width of a twentieth of the cutoff.
    """
    beta = 20.0 / kcut if beta is None else beta

    def w(*k):
        out = 1.0
        for a in k:
            out = out * expit(-beta * (np.abs(np.asarray(a, dtype=np.float64)) - kcut))
        return out
    return w


# --------------------------------------------------------------------------
# Schrodinger family
# --------------------------------------------------------------------------

def schrodinger(V=None, m: float = 1.0, hbar: float = 1.0, kcut: float | None = None,
                beta: float | None = None) -> Equation:
    """``i hbar d_t psi = -(hbar**2 / 2m) lap psi + V(x) psi``."""
    return Equation(linear=lambda *k: -1j * hbar * k2(*k) / (2 * m),
                    potential=None if V is None else (lambda *x: -1j * V(*x) / hbar),
                    filter=None if kcut is None else fermi_lowpass(kcut, beta),
                    name="Schrodinger")


def gross_pitaevskii(V=None, g: float = 0.0, m: float = 1.0, kcut: float | None = None,
                     beta: float | None = None, imaginary_time: bool = False) -> Equation:
    """``i d_t psi = -(1/2m) lap psi + V psi + g |psi|**2 psi`` (``hbar = 1``).

    The nonlinear step is exact: ``|psi|`` is constant along the local flow, so it is
    the phase rotation ``exp(-i g |psi|**2 h) psi``.

    ``imaginary_time=True`` gives ``d_tau psi = -H psi`` with renormalisation after
    every step, which relaxes any initial state onto the ground state of ``H``
    (including the ``g |psi|**2`` mean field). Its local flow is also exact:
    ``|psi|**2`` obeys ``d|psi|**2/dtau = -2g|psi|**4``.
    """
    if imaginary_time:
        return Equation(
            linear=lambda *k: -k2(*k) / (2 * m),
            potential=None if V is None else (lambda *x: -V(*x)),
            flow=None if g == 0 else (lambda u, *xt: u / np.sqrt(1.0 + 2.0 * g * np.abs(u) ** 2 * xt[-1])),
            normalise=True, name="Gross-Pitaevskii (imaginary time)")
    return Equation(
        linear=lambda *k: -1j * k2(*k) / (2 * m),
        potential=None if V is None else (lambda *x: -1j * V(*x)),
        flow=None if g == 0 else (lambda u, *xt: np.exp(-1j * g * np.abs(u) ** 2 * xt[-1]) * u),
        filter=None if kcut is None else fermi_lowpass(kcut, beta),
        name="Gross-Pitaevskii")


def nonlinear_schrodinger(f, V=None, m: float = 1.0, kcut: float | None = None,
                          beta: float | None = None) -> Equation:
    """``i d_t psi = -(1/2m) lap psi + V psi + f(|psi|**2) psi`` for any real ``f``:
    the local step is the exact phase rotation ``exp(-i f(|psi|**2) h)``."""
    return Equation(
        linear=lambda *k: -1j * k2(*k) / (2 * m),
        potential=None if V is None else (lambda *x: -1j * V(*x)),
        flow=lambda u, *xt: np.exp(-1j * f(np.abs(u) ** 2) * xt[-1]) * u,
        filter=None if kcut is None else fermi_lowpass(kcut, beta),
        name="nonlinear Schrodinger")


def coupled_gross_pitaevskii(G, V=None, m: float = 1.0, kcut: float | None = None,
                             beta: float | None = None, imaginary_time: bool = False,
                             rk_substeps: int = 1) -> Equation:
    """A multi-component condensate, ``i d_t psi_a = -(1/2m) lap psi_a + V_a psi_a
    + sum_b G[a, b] |psi_b|**2 psi_a``.

    ``G`` is the interaction matrix; ``V`` one potential shared by all components or a
    sequence with one per component. Every ``|psi_b|`` is constant along the local
    flow, so the nonlinear step is the exact phase rotation for each component.

    ``imaginary_time=True`` relaxes towards the ground state with *every component's
    norm held fixed* (``normalise="each"``) -- a mixture with given populations. The
    densities then obey a coupled Lotka-Volterra system with no closed form, so that
    local step is integrated by RK4.
    """
    G = np.asarray(G, dtype=np.float64)
    n = G.shape[0]
    unit = 1.0 if imaginary_time else 1j

    if V is None:
        pot = None
    elif callable(V):
        pot = lambda *x: -unit * V(*x)
    else:
        pot = [None if Va is None else (lambda *x, Va=Va: -unit * Va(*x)) for Va in V]
    linear = lambda *k: -unit * k2(*k) / (2 * m)

    if imaginary_time:
        def rhs(u, *xt):
            dens = [np.abs(a) ** 2 for a in u]
            return tuple(-sum(G[a, b] * dens[b] for b in range(n)) * u[a] for a in range(n))
        return Equation(linear=linear, potential=pot, nonlinear=rhs, ncomp=n,
                        normalise="each", rk_substeps=rk_substeps,
                        name=f"{n}-component Gross-Pitaevskii (imaginary time)")

    def flow(u, *xt):
        h = xt[-1]
        dens = [np.abs(a) ** 2 for a in u]
        return tuple(np.exp(-1j * h * sum(G[a, b] * dens[b] for b in range(n))) * u[a]
                     for a in range(n))
    return Equation(linear=linear, potential=pot, flow=flow, ncomp=n,
                    filter=None if kcut is None else fermi_lowpass(kcut, beta),
                    name=f"{n}-component Gross-Pitaevskii")


# --------------------------------------------------------------------------
# diffusion and reaction-diffusion
# --------------------------------------------------------------------------

def heat(D: float = 1.0, source=None, rate=None) -> Equation:
    """``d_t u = D lap u + rate(x) u + source(u, x, t)``."""
    return Equation(linear=lambda *k: -D * k2(*k), potential=rate, nonlinear=source,
                    name="heat")


def fractional_diffusion(alpha: float, D: float = 1.0) -> Equation:
    """``d_t u = -D (-lap)**(alpha/2) u`` -- Levy flights for ``alpha < 2``."""
    return Equation(linear=lambda *k: -D * k2(*k) ** (alpha / 2), name="fractional diffusion")


def fisher_kpp(D: float = 1.0, r: float = 1.0) -> Equation:
    """``d_t u = D lap u + r u (1 - u)``. Local flow: the logistic map
    ``u(t+h) = u / (u + (1 - u) exp(-r h))``."""
    return Equation(linear=lambda *k: -D * k2(*k),
                    flow=lambda u, *xt: u / (u + (1.0 - u) * np.exp(-r * xt[-1])),
                    name="Fisher-KPP")


def allen_cahn(eps: float = 1.0) -> Equation:
    """``d_t u = eps lap u + u - u**3``. Local flow:
    ``u(t+h) = u / sqrt(u**2 + (1 - u**2) exp(-2h))``."""
    return Equation(linear=lambda *k: -eps * k2(*k),
                    flow=lambda u, *xt: u / np.sqrt(u**2 + (1.0 - u**2) * np.exp(-2.0 * xt[-1])),
                    name="Allen-Cahn")


def complex_ginzburg_landau(b: float = 0.0, c: float = 0.0) -> Equation:
    """``d_t A = A + (1 + i b) lap A - (1 + i c) |A|**2 A``. Local flow:
    ``A(t+h) = A (1 + 2|A|**2 h)**(-(1 + i c)/2)``."""
    return Equation(linear=lambda *k: 1.0 - (1.0 + 1j * b) * k2(*k),
                    flow=lambda u, *xt: u * (1.0 + 2.0 * np.abs(u) ** 2 * xt[-1]) ** (-(1.0 + 1j * c) / 2),
                    name="complex Ginzburg-Landau")


def gray_scott(Du: float = 2e-5, Dv: float = 1e-5, F: float = 0.04, k: float = 0.06,
               rk_substeps: int = 1) -> Equation:
    """The two-species Gray-Scott system,

        d_t u = Du lap u - u v**2 + F (1 - u)
        d_t v = Dv lap v + u v**2 - (F + k) v.

    The local ODE has no closed form, so it is integrated by RK4."""
    def rhs(uv, *xt):
        u, v = uv
        return (-u * v**2 + F * (1.0 - u), u * v**2 - (F + k) * v)
    return Equation(linear=[lambda *q: -Du * k2(*q), lambda *q: -Dv * k2(*q)],
                    nonlinear=rhs, ncomp=2, rk_substeps=rk_substeps, name="Gray-Scott")


def fitzhugh_nagumo(Du: float = 1.0, Dv: float = 0.0, a: float = 0.1, eps: float = 0.01,
                    gamma: float = 1.0, rk_substeps: int = 1) -> Equation:
    """``d_t u = Du lap u + u (1 - u)(u - a) - v``,
    ``d_t v = Dv lap v + eps (u - gamma v)``. RK4 for the local part."""
    def rhs(uv, *xt):
        u, v = uv
        return (u * (1.0 - u) * (u - a) - v, eps * (u - gamma * v))
    return Equation(linear=[lambda *q: -Du * k2(*q), lambda *q: -Dv * k2(*q)],
                    nonlinear=rhs, ncomp=2, rk_substeps=rk_substeps, name="FitzHugh-Nagumo")


# --------------------------------------------------------------------------
# linear transport
# --------------------------------------------------------------------------

def advection_diffusion(velocity, D: float = 0.0) -> Equation:
    """``d_t u = -(c . grad) u + D lap u`` for a constant velocity ``c`` (scalar in 1D,
    a vector otherwise). The advection is exact: it is the multiplier ``-i c . k``."""
    c = np.atleast_1d(np.asarray(velocity, dtype=np.float64))
    return Equation(linear=lambda *k: -1j * sum(ci * ki for ci, ki in zip(c, k)) - D * k2(*k),
                    name="advection-diffusion")
