"""Gates for the general split-step solver (``tnde.pde``).

Each test pins one layer against something independent of the tensor-train code:
analytic solutions where they exist (Gaussian diffusion, the free Schrodinger packet,
the harmonic-oscillator ground state, a time-dependent local ODE), the dense
``tnde.reference`` solver otherwise -- which shares the stepping loop and the local
flow with the train solver, so a disagreement isolates the train machinery.

The 2D and 3D cases are deliberately *anisotropic* (unequal box lengths, unequal
velocities): the Fourier transform of an interleaved train returns the variables in
reverse order, and a symmetric test would pass with that order handled wrongly.
"""
import sys

import numpy as np

sys.path.insert(0, "/home/joselado/Documents/programs/tnde")
from tnde import equations, pde, reference, tt
from tnde.grid import Grid
from tnde.pde import Equation


def relerr(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def test_grid_round_trip():
    g = Grid(5, [(-4.0, 4.0), (0.0, 12.0), (-1.0, 1.0)])
    rng = np.random.default_rng(1)
    ij = [rng.integers(0, g.M, 300) for _ in range(3)]
    back = g.split(g.fuse(*ij))
    assert all(np.array_equal(a, b) for a, b in zip(ij, back))
    x = g.coords(g.fuse(*ij))
    assert np.allclose(x[1], 0.0 + ij[1] * 12.0 / 32)
    # the nearest grid point to a coordinate is that coordinate's own index
    assert np.array_equal(g.nearest(*x), g.fuse(*ij))
    # momenta: index M-1 is -2 pi / L, index 1 is +2 pi / L
    k = g.momenta(g.fuse(np.array([1, g.M - 1]), np.zeros(2, int), np.zeros(2, int)))
    assert np.allclose(k[0], [2 * np.pi / 8, -2 * np.pi / 8])
    print(f"  fuse/split, coords, nearest, momenta on {g}: ok")


def test_interpolate_integral_normalise():
    g = Grid(7, [(-8.0, 8.0), (-8.0, 8.0)])
    f = lambda x, y: np.exp(-((x - 0.5) ** 2 + 2 * (y + 0.3) ** 2) / 2)
    u = pde.interpolate(f, g, tolerance=1e-12)
    e = pde.max_error(u, f, g, n=64)
    I = pde.integral(u, g).real
    exact = 2 * np.pi / np.sqrt(2)
    n = pde.norm(pde.normalise(u, g), g)
    v = pde.evaluate(u, g, 0.5, -0.3)                 # snaps to the nearest grid point
    want = f(*g.coords(g.nearest(0.5, -0.3)))
    print(f"  interpolation error {e:.2e}, integral {I:.8f} (exact {exact:.8f}), "
          f"norm after normalise {n:.12f}, point value {abs(v):.6f} (exact {want:.6f})")
    assert e < 1e-10 and abs(I - exact) < 1e-6 and abs(n - 1) < 1e-12
    assert abs(v - want) < 1e-10


def test_exact_flows_match_rk4():
    """Every closed-form local flow in ``tnde.equations`` must agree with RK4
    integration of the same right-hand side -- this is what validates the algebra."""
    rng = np.random.default_rng(0)
    u = 0.9 * rng.random(200) + 0.05 + 0.3j * rng.standard_normal(200)
    x = (rng.random(200),)
    h = 0.3
    cases = {
        "Fisher-KPP": (equations.fisher_kpp(r=1.3), lambda u, *xt: 1.3 * u * (1 - u)),
        "Allen-Cahn": (equations.allen_cahn(), lambda u, *xt: u - u**3),
        "CGL": (equations.complex_ginzburg_landau(b=0.5, c=-1.2),
                lambda u, *xt: -(1 + 1j * (-1.2)) * np.abs(u) ** 2 * u),
        "GP": (equations.gross_pitaevskii(g=2.0), lambda u, *xt: -2j * np.abs(u) ** 2 * u),
        "GP imaginary": (equations.gross_pitaevskii(g=2.0, imaginary_time=True),
                         lambda u, *xt: -2.0 * np.abs(u) ** 2 * u),
        "NLS": (equations.nonlinear_schrodinger(lambda d: np.log1p(d)),
                lambda u, *xt: -1j * np.log1p(np.abs(u) ** 2) * u),
    }
    for name, (eq, rhs) in cases.items():
        ref = Equation(nonlinear=rhs, rk_substeps=400)
        got = eq.local_flow([u], x, 0.0, h)[0]
        want = ref.local_flow([u], x, 0.0, h)[0]
        e = relerr(got, want)
        print(f"  {name:<13} exact flow vs RK4(400 substeps): {e:.2e}")
        assert e < 1e-9
    G = np.array([[1.0, 0.5], [0.5, 2.0]])
    eq = equations.coupled_gross_pitaevskii(G)
    ref = Equation(nonlinear=lambda uv, *xt: tuple(
        -1j * sum(G[a, b] * np.abs(uv[b]) ** 2 for b in range(2)) * uv[a] for a in range(2)),
        ncomp=2, rk_substeps=400)
    v = [u, u[::-1] * 0.7]
    got, want = eq.local_flow(v, x, 0.0, h), ref.local_flow(v, x, 0.0, h)
    e = max(relerr(a, b) for a, b in zip(got, want))
    print(f"  {'coupled GP':<13} exact flow vs RK4(400 substeps): {e:.2e}")
    assert e < 1e-9


def test_heat_1d_analytic():
    g = Grid(10, (-20.0, 20.0))
    D, s0, dt, n = 0.5, 1.0, 0.05, 40
    u, _ = pde.solve(equations.heat(D), lambda x: np.exp(-x**2 / (2 * s0**2)), g, dt, n,
                     tolerance=1e-12, maxdim=64, verbose=False)
    s2 = s0**2 + 2 * D * n * dt
    X, = g.mesh()
    e = relerr(pde.to_dense(u, g).real, s0 / np.sqrt(s2) * np.exp(-X**2 / (2 * s2)))
    print(f"  Gaussian spreading, {n} steps: error vs analytic {e:.2e}, rank {u.rank()}")
    assert e < 1e-10


def test_free_schrodinger_analytic():
    """A free Gaussian packet with momentum k0: exact dispersion and translation."""
    g = Grid(11, (-40.0, 40.0))
    k0, dt, n = 1.5, 0.02, 100
    psi0 = lambda x: (1 / np.pi) ** 0.25 * np.exp(-x**2 / 2 + 1j * k0 * x)
    u, _ = pde.solve(equations.schrodinger(), psi0, g, dt, n, tolerance=1e-12, maxdim=64,
                     verbose=False)
    T = n * dt
    X, = g.mesh()
    exact = ((1 / np.pi) ** 0.25 / np.sqrt(1 + 1j * T)
             * np.exp(-(X - k0 * T) ** 2 / (2 * (1 + 1j * T)) + 1j * k0 * X - 1j * k0**2 * T / 2))
    e = relerr(pde.to_dense(u, g), exact)
    print(f"  t = {T}: error vs analytic packet {e:.2e}, rank {u.rank()}, "
          f"norm {pde.norm(u, g):.10f}")
    assert e < 1e-8


def test_gp_snapshots_match_dense_and_unmerged():
    """Potential + nonlinearity + momentum window, with snapshots. The snapshots
    must be the true Strang states (the merged loop pays an extra half-step for
    them), so a run that snapshots every step -- and therefore never merges -- must
    give the same result as one that merges throughout."""
    g = Grid(10, (-16.0, 16.0))
    eq = equations.gross_pitaevskii(V=lambda x: 0.5 * x**2, g=3.0, kcut=20.0)
    psi0 = lambda x: np.exp(-(x - 1.0) ** 2 / 2) * np.exp(0.5j * x)
    dt, n = 0.02, 20
    u, snaps = pde.solve(eq, psi0, g, dt, n, tolerance=1e-12, maxdim=64, save_every=10,
                         verbose=False)
    ud, snapsd = reference.solve(eq, psi0, g, dt, n, save_every=10)
    for j in sorted(snaps):
        e = relerr(pde.to_dense(snaps[j], g), snapsd[j])
        print(f"  step {j:>2}: TT vs dense {e:.2e}, rank {snaps[j].rank()}")
        assert e < 1e-9
    u1, _ = pde.solve(eq, psi0, g, dt, n, tolerance=1e-12, maxdim=64, save_every=1,
                      verbose=False)
    e = relerr(pde.to_dense(u1, g), pde.to_dense(u, g))
    print(f"  merged vs unmerged half-steps: {e:.2e}")
    assert e < 1e-9


def test_time_dependent_local_term():
    """``d_t u = i a x cos(w t) u`` has the exact solution ``u0 exp(i a x sin(wt)/w)``.
    Integrated by RK4 as a `nonlinear` term, this checks that the merged loop hands
    every sub-step the right starting time: an offset of dt/2 would show as O(dt)."""
    g = Grid(8, (-2.0, 2.0))
    a, w, dt, n = 1.0, 3.0, 0.01, 100
    eq = Equation(nonlinear=lambda u, x, t: 1j * a * x * np.cos(w * t) * u)
    u0 = lambda x: np.exp(-x**2)
    u, _ = pde.solve(eq, u0, g, dt, n, tolerance=1e-12, maxdim=64, save_every=30, verbose=False)
    X, = g.mesh()
    exact = u0(X) * np.exp(1j * a * X * np.sin(w * n * dt) / w)
    e = relerr(pde.to_dense(u, g), exact)
    print(f"  driven phase, RK4 through merged steps: error {e:.2e} (O(dt**4) ~ 1e-8)")
    assert e < 1e-7


def test_2d_anisotropic_matches_dense():
    g = Grid(5, [(-4.0, 4.0), (-6.0, 6.0)])
    eq = equations.advection_diffusion([1.0, -0.5], D=0.1)
    u0 = lambda x, y: np.exp(-(x**2 + 0.5 * y**2))
    u, _ = pde.solve(eq, u0, g, 0.05, 10, tolerance=1e-12, maxdim=200, verbose=False)
    ud, _ = reference.solve(eq, u0, g, 0.05, 10)
    d = pde.to_dense(u, g)
    e, et = relerr(d, ud), relerr(d, ud.T)
    print(f"  2D advection-diffusion, unequal box and velocities: TT vs dense {e:.2e} "
          f"(vs transposed dense {et:.2e})")
    assert e < 1e-10 and et > 0.1


def test_3d_heat_matches_dense():
    g = Grid(4, [(-3.0, 3.0), (-4.0, 4.0), (-5.0, 5.0)])
    u0 = lambda x, y, z: np.exp(-(x**2 + 0.7 * y**2 + 0.4 * z**2)) * (1 + 0.3 * x - 0.2 * z)
    u, _ = pde.solve(equations.heat(0.2), u0, g, 0.05, 5, tolerance=1e-12, maxdim=200,
                     verbose=False)
    ud, _ = reference.solve(equations.heat(0.2), u0, g, 0.05, 5)
    e = relerr(pde.to_dense(u, g), ud)
    print(f"  3D heat on a {g.M}^3 box: TT vs dense {e:.2e}, rank {u.rank()}")
    assert e < 1e-10


def test_fractional_and_hyper_diffusion():
    g = Grid(9, (-10.0, 10.0))
    u0 = lambda x: np.exp(-x**2) * (1 + 0.5 * np.cos(2 * x))
    for name, eq in (("fractional alpha=1.2", equations.fractional_diffusion(1.2, D=0.3)),
                     ("hyperviscous k**4", Equation(linear=lambda k: -0.01 * k**4))):
        u, _ = pde.solve(eq, u0, g, 0.05, 10, tolerance=1e-12, maxdim=64, verbose=False)
        ud, _ = reference.solve(eq, u0, g, 0.05, 10)
        e = relerr(pde.to_dense(u, g), ud)
        print(f"  {name}: TT vs dense {e:.2e}")
        assert e < 1e-10


def test_two_component_systems():
    g = Grid(8, (0.0, 1.0))
    eq = equations.gray_scott(Du=2e-5, Dv=1e-5, F=0.04, k=0.06, rk_substeps=2)
    bump = lambda x: np.exp(-((x - 0.5) / 0.05) ** 2)
    u0 = [lambda x: 1.0 - 0.5 * bump(x), lambda x: 0.25 * bump(x)]
    u, _ = pde.solve(eq, u0, g, 1.0, 20, tolerance=1e-12, maxdim=64, verbose=False)
    ud, _ = reference.solve(eq, u0, g, 1.0, 20)
    e = max(relerr(pde.to_dense(a, g), b) for a, b in zip(u, ud))
    print(f"  Gray-Scott (RK4): TT vs dense {e:.2e}, ranks {[c.rank() for c in u]}")
    assert e < 1e-9

    g = Grid(9, (-12.0, 12.0))
    eq = equations.coupled_gross_pitaevskii([[1.0, 0.8], [0.8, 1.0]],
                                            V=lambda x: 0.5 * x**2, kcut=12.0)
    u0 = [lambda x: np.exp(-x**2 / 2), lambda x: 0.6 * np.exp(-(x - 1.0) ** 2 / 2)]
    u, _ = pde.solve(eq, u0, g, 0.02, 15, tolerance=1e-12, maxdim=64, verbose=False)
    ud, _ = reference.solve(eq, u0, g, 0.02, 15)
    e = max(relerr(pde.to_dense(a, g), b) for a, b in zip(u, ud))
    print(f"  coupled GP (exact flow): TT vs dense {e:.2e}, ranks {[c.rank() for c in u]}")
    assert e < 1e-9

    # immiscible mixture in imaginary time: each component keeps its own norm
    eq = equations.coupled_gross_pitaevskii(5 * np.array([[1.0, 1.5], [1.5, 1.0]]),
                                            V=lambda x: 0.5 * x**2, imaginary_time=True)
    u0 = [lambda x: np.exp(-(x - 0.3) ** 2 / 4), lambda x: np.exp(-(x + 0.3) ** 2 / 4)]
    u, _ = pde.solve(eq, u0, g, 0.02, 30, tolerance=1e-12, maxdim=64, verbose=False)
    ud, _ = reference.solve(eq, u0, g, 0.02, 30)
    e = max(relerr(pde.to_dense(a, g), b) for a, b in zip(u, ud))
    norms = [pde.norm(c, g) for c in u]
    print(f"  coupled GP, imaginary time (RK4, normalise='each'): TT vs dense {e:.2e}, "
          f"norms {norms[0]:.10f} {norms[1]:.10f}")
    assert e < 1e-9 and all(abs(n - 1) < 1e-10 for n in norms)


def test_imaginary_time_ground_state():
    """Imaginary-time relaxation onto the harmonic-oscillator ground state, from a
    displaced, wider Gaussian. Second order in dt, so the residual is the splitting
    error at dt = 0.05 plus the exp(-tau) tail of the first excited state."""
    g = Grid(10, (-12.0, 12.0))
    eq = equations.gross_pitaevskii(V=lambda x: 0.5 * x**2, imaginary_time=True)
    u, _ = pde.solve(eq, lambda x: np.exp(-(x - 2.0) ** 2 / 8), g, 0.05, 200,
                     tolerance=1e-12, maxdim=64, verbose=False)
    X, = g.mesh()
    got = pde.to_dense(u, g)
    got = got * np.sign(got.real[g.M // 2])
    e = relerr(got, (1 / np.pi) ** 0.25 * np.exp(-X**2 / 2))
    print(f"  harmonic-oscillator ground state after tau = 10: error {e:.2e}, "
          f"norm {pde.norm(u, g):.12f}")
    assert e < 1e-3
    # with interactions there is no closed form: pin against the dense reference
    eq = equations.gross_pitaevskii(V=lambda x: 0.5 * x**2, g=5.0, imaginary_time=True)
    u, _ = pde.solve(eq, lambda x: np.exp(-x**2 / 2), g, 0.05, 60, tolerance=1e-12,
                     maxdim=64, verbose=False)
    ud, _ = reference.solve(eq, lambda x: np.exp(-x**2 / 2), g, 0.05, 60)
    e = relerr(pde.to_dense(u, g), ud)
    print(f"  interacting (g = 5) relaxation: TT vs dense {e:.2e}, rank {u.rank()}")
    assert e < 1e-9


if __name__ == "__main__":
    for t in (test_grid_round_trip, test_interpolate_integral_normalise,
              test_exact_flows_match_rk4, test_heat_1d_analytic,
              test_free_schrodinger_analytic, test_gp_snapshots_match_dense_and_unmerged,
              test_time_dependent_local_term, test_2d_anisotropic_matches_dense,
              test_3d_heat_matches_dense, test_fractional_and_hyper_diffusion,
              test_two_component_systems, test_imaginary_time_ground_state):
        print(f"{t.__name__}:")
        t()
    print("PASS")
