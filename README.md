# tnde — differential equations on tensor networks

`tnde` evolves time-dependent partial differential equations on grids far too fine to
store, by representing the field as a **quantics tensor train**: a 2³⁰-point line, a
2²⁰ × 2²⁰ plane or a 2¹⁵ × 2¹⁵ × 2¹⁵ box becomes a chain of 30, 40 or 45 small
tensors, every operator becomes a matrix product operator, and the nonlinear terms are
re-fitted each step by tensor cross interpolation from a few thousand function
evaluations. The grid never exists in memory; the cost is set by how much structure
the solution has (its bond dimension), not by how many points it has.

It solves, in one, two or three space dimensions and for scalar or multi-component
fields,

$$\partial_t u \;=\; \ell(-i\nabla)\,u \;+\; p(x)\,u \;+\; N(u, x, t),$$

where

* **$\ell$ is any Fourier multiplier** — the symbol of a linear, translation-invariant
  operator: $-ik^2/2m$ (Schrödinger), $-Dk^2$ (diffusion), $-\nu k^4$
  (hyperviscosity), $-|k|^\alpha$ (a fractional Laplacian), $-i\,c\cdot k$
  (advection), or anything else you can write down;
* **$p$ is a local linear term** — a potential, $-iV(x)$ in real time, $-V(x)$ in
  imaginary time, a spatially varying growth rate;
* **$N$ is a local nonlinearity** — at each point an ODE in $u$ alone, such as
  $-ig|u|^2u$, $ru(1-u)$, $u-u^3$, or any coupling between components.

Ready-made equations (`tnde.equations`):

| | equation |
|---|---|
| `schrodinger`, `gross_pitaevskii`, `nonlinear_schrodinger` | $i\partial_t\psi = -\tfrac{1}{2m}\nabla^2\psi + V\psi + f(\lvert\psi\rvert^2)\psi$, real or imaginary time |
| `coupled_gross_pitaevskii` | a multi-component condensate with interaction matrix $G_{ab}$ |
| `heat`, `fractional_diffusion`, `advection_diffusion` | linear transport |
| `fisher_kpp`, `allen_cahn` | scalar reaction–diffusion |
| `complex_ginzburg_landau` | $\partial_t A = A + (1+ib)\nabla^2A - (1+ic)\lvert A\rvert^2A$ |
| `gray_scott`, `fitzhugh_nagumo` | two-species reaction–diffusion |

Everything is built from three callables, so an equation not on the list is one
`Equation(linear=..., potential=..., nonlinear=...)` away.

What is *not* covered: a nonlinearity containing derivatives ($u\,u_x$ in Burgers or
KdV). Such a term is not local, and the splitting below does not apply to it.

## Quick start

```python
import numpy as np
from tnde import Grid, equations, pde

# ground state of a 1D Bose gas: imaginary-time Gross-Pitaevskii on 2**20 points
grid = Grid(20, (-16.0, 16.0))
eq = equations.gross_pitaevskii(V=lambda x: 0.5 * x**2, g=50.0, imaginary_time=True)
psi, snapshots = pde.solve(eq, lambda x: np.exp(-x**2 / 8), grid, dt=0.01, nsteps=400,
                           tolerance=1e-10, maxdim=30, save_every=100)

(x,), values = pde.reconstruct(psi, grid, n=1024)   # sample the field on 1024 points
n0 = abs(pde.evaluate(psi, grid, 0.0)) ** 2         # the density at one point
```

A 2D equation takes a 2D grid and functions of two coordinates, nothing else changes:

```python
grid = Grid(12, [(-20.0, 20.0), (-20.0, 20.0)])              # 4096 x 4096 = 1.7e7 points
eq = equations.advection_diffusion(velocity=(1.0, -0.6), D=0.05)
u, snaps = pde.solve(eq, lambda x, y: np.exp(-x**2 / 0.98 - y**2 / 2.42), grid,
                     dt=0.1, nsteps=50, tolerance=1e-8, maxdim=40)
```

Your own equation is the three maps. Time is the last argument of the nonlinear term,
after the coordinates; a multi-component field passes and returns tuples:

```python
from tnde import Equation

# 1D: d_t u = -0.01 k**4 u  +  (-x**2) u  +  i a x cos(w t) u
eq = Equation(linear=lambda k: -0.01 * k**4,
              potential=lambda x: -x**2,
              nonlinear=lambda u, x, t: 1j * a * x * np.cos(w * t) * u)

# 2 components, 2D, with an exact solution of the local ODE instead of RK4
eq = Equation(linear=[lambda kx, ky: -Du * (kx**2 + ky**2), lambda kx, ky: -Dv * (kx**2 + ky**2)],
              flow=lambda uv, x, y, t, h: my_exact_local_step(uv, h), ncomp=2)
```

Worked examples, each checked against a dense reference:

| script | what |
|---|---|
| `examples/ground_state_gp.py` | 1D Bose gas ground state by imaginary time, against Thomas–Fermi |
| `examples/binary_mixture_1d.py` | an immiscible two-component condensate separating, populations held fixed |
| `examples/advection_diffusion_2d.py` | a Gaussian on a 4096² grid against the exact solution — the case the representation is for |
| `examples/allen_cahn_droplet_2d.py` | a droplet shrinking by curvature flow — the case it is *not* for, kept to show the cost |
| `examples/reproduce_paper_1d.py`, `reproduce_paper_2d.py` | the Gross–Pitaevskii benchmarks (see below) |

## How it works

**Representation.** A function on $2^R$ points is indexed by $R$ bits, and a smooth
function is a tensor train over those bits with a small bond dimension — 10–20 for
the fields here. In $d$ dimensions the bits of all variables are interleaved by length
scale, $x_1y_1x_2y_2\ldots$, so the train has $dR$ sites. The mapping between grid
indices, coordinates and momenta lives in `Grid`.

**Splitting.** Each step of length $h$ applies

$$N(h/2)\;P(h/2)\;L(h)\;P(h/2)\;N(h/2),$$

second-order Strang splitting of three operators. Consecutive $N(h/2)$ factors of
neighbouring steps are merged into one $N(h)$ wherever nothing needs the true state at
a step boundary, so a long run costs one nonlinear application per step; snapshots
and renormalisations pay the extra half-step where they occur.

* $L(h)$ is exact: a quantics Fourier transform (an analytic MPO of rank ≤ 12 applied
  to each variable's sites), multiplication by $e^{h\ell(k)}$ as a diagonal MPO, and the
  inverse transform.
* $P(h/2)$ is exact: the diagonal MPO of $e^{h p(x)/2}$, cross-interpolated once.
* $N(h)$ is applied by **re-interpolation**: wherever TCI asks for a value, the current
  train is evaluated, the local ODE is advanced there, and the result is fitted into a
  new train. The local ODE is solved by a closed-form flow map when the equation
  supplies one (every preset that has one does) and by classical RK4 otherwise. This
  is the step no fixed operator can express, and it dominates the cost.

**Momentum windows.** A diffusive multiplier $e^{-hDk^2}$ decays and is low-rank. An
oscillatory one, $e^{-ihk^2/2m}$, is not: at the largest momentum of a fine grid its
phase changes by many radians per index and no small train represents it. Schrödinger-type
presets therefore accept a cutoff `kcut`, applied as a separable Fermi–Dirac window
$w(k)$ (`equations.fermi_lowpass`). It is a rank-control device with a physical cost —
momentum content above the cutoff is discarded, and the norm afterwards is a
diagnostic only if the window is known to sit above everything the dynamics excites.

**Operator application.** A diagonal MPO of rank $r$ applied to a state of rank $\chi$
has an exact product of rank $r\chi$, which is then truncated. That is cheap and slightly
more accurate when both are small (1D) and prohibitive when both are 50–70 (2D, 3D),
where a two-site variational fit at fixed bond dimension is ~50× faster. `solve` picks by
dimension unless told otherwise; the Fourier MPOs, being non-local, always go through the
exact product.

**Reference.** `tnde.reference` runs the *identical* scheme — the same stepping loop,
the same local flow — on dense arrays with `numpy.fft`. Every difference between it
and the train solver is truncation, which is what the tests isolate.

## Validation

`./run_tests.sh` runs the gates; `tests/test_pde.py` covers the general solver and
pins every layer against something independent of the tensor-train code:

| gate | reference | result |
|---|---|---|
| Gaussian diffusion, 1D, 40 steps | analytic | 9e-13 |
| free Schrödinger packet, 1D, $t=2$ | analytic | 5e-12 |
| harmonic-oscillator ground state by imaginary time | analytic | 1e-4 (the $O(\Delta t^2)$ splitting error) |
| driven local term $\partial_t u = iax\cos(\omega t)u$ through merged steps | analytic | 6e-12 |
| every closed-form local flow (Fisher, Allen–Cahn, CGL, GP, coupled GP, …) | RK4 with 400 sub-steps | ≤ 3e-12 |
| Gross–Pitaevskii with trap, nonlinearity and window, with snapshots | dense | 4e-12 |
| merged vs. unmerged half-steps | each other | 3e-13 |
| anisotropic 2D advection–diffusion (unequal box, unequal velocities) | dense | 9e-14 (transposed: 0.7) |
| 3D heat on an anisotropic box | dense | 8e-14 |
| fractional and hyperviscous diffusion | dense | ≤ 2e-12 |
| Gray–Scott (RK4) and coupled GP (exact flow), two components | dense | ≤ 8e-12 |
| binary mixture in imaginary time, per-component norms | dense | 2e-12 |

The 2D and 3D cases are deliberately anisotropic. The Fourier transform of an
interleaved train returns the variables in *reverse* order — each variable's transform
reverses its own digits, and the whole-train reversal that restores bit order also
swaps the slots — and a symmetric test would pass with that handled wrongly.

## What it costs

One core, single-threaded BLAS, the example scripts as committed:

| example | grid | per step | result |
|---|---|---|---|
| `ground_state_gp.py`: 1D Bose gas, $g = 50$, imaginary time | 2¹⁶ | 0.17 s | $\mu$ within 1.5 % of Thomas–Fermi (the kinetic correction); matches the dense reference to 4e-10 |
| `binary_mixture_1d.py`: two components, imaginary time, RK4 local step | 2¹⁴ | 0.36 s | species separate to $\langle x\rangle = \pm1.50$, overlap 0.26 → 0.02; dense reference 8e-9 |
| `advection_diffusion_2d.py`: drifting, spreading Gaussian | 4096² | 2.1 s | ≤ 2e-7 against the exact solution over 50 steps, rank 40 |
| `allen_cahn_droplet_2d.py`: curvature-driven shrinking | 128² | 1.0 s | radius follows $r^2 = r_0^2 - 2\varepsilon t$ to 0.5 %; rank 50 at tolerance 1e-5 |
| `reproduce_paper_1d.py`: Gross–Pitaevskii in a lattice | 2³⁰ | 3 s | reproduces the published tensors (below) |
| `reproduce_paper_2d.py`: Gross–Pitaevskii in a 2D quasi-crystal | 2²⁰ × 2²⁰ | 73 s | — |

Two things set the cost, and neither is the number of grid points.

* **The rank of the field.** A smooth or nearly separable field is rank 10–20 in 1D. In
  2D and 3D the interleaved train has *multiplicative* rank for separable structure — a
  Gaussian times a Gaussian is $r_x r_y$ at the cross bonds — so 40–70 is normal, and the
  solver switches to the variational fit for its operator applications there
  (`method='auto'`). A sharp front that is curved, or a field with many independent
  features, has no small train at all: the droplet above is rank 90 at tolerance 1e-8
  and rank 50 at 1e-5, and every step costs seconds on a grid `numpy` would finish in
  milliseconds. The quantics representation is for fields whose structure is small
  compared with their grid, not for fields that need the grid.
* **The nonlinear step.** It is a cross interpolation from scratch every step, with
  the field evaluated at every point TCI asks for. Its cost is a few thousand evaluations
  times the rank squared; an exact flow map costs one evaluation per point and RK4 four
  per sub-step. This is where the time goes in every nonlinear run above.


## The Gross–Pitaevskii solvers

The library grew out of a 1D/2D Gross–Pitaevskii solver, and those drivers are kept
as they are: `tnde.evolve1d`, `tnde.evolve2d`, their dense oracle `tnde.dense` and
`tnde.observables`. They reproduce a published quantics-tensor-train benchmark
([arXiv:2507.04262](https://arxiv.org/abs/2507.04262)) tensor for tensor — at $R=30$,
2³⁰ points on $[-500, 500]$, a Gaussian in a harmonic trap plus a fast lattice, $g=5$,
$\Delta t = 0.01$, bond dimension 14:

| step | rank | ref rank | 1 − overlap | max pointwise rel. diff |
|---|---|---|---|---|
| 1 | 10 | 10 | 0.0 | 8.4e-11 |
| 10 | 14 | 14 | 3.0e-13 | 1.7e-07 |
| 50 | 14 | 14 | 9.8e-11 | 3.3e-06 |
| 100 | 14 | 14 | 2.2e-10 | 7.7e-06 |

at 3 s per step, bond dimensions matching exactly throughout
(`tests/test_reference_r30.py`, `examples/reproduce_paper_1d.py`). The 2D driver runs
the benchmark's $2^{20}\times2^{20}$ configuration at 73 s per step.

Those drivers keep the benchmark's conventions, which differ from the general
solver's: a grid that includes the endpoint, the finite-difference dispersion
$-\tfrac{4M^2}{L^2}\sin^2(\pi k/M)$ rather than $-k^2$, a renormalisation after every
kinetic step, and a momentum cutoff searched on a Gaussian trial state. Use them to
reproduce that benchmark; use `tnde.pde` for everything else. `docs/guide.tex` is
their physics guide — units, what a step means on the time axis, the momentum map,
and an error budget whose main finding is that **the momentum window, not the bond
dimension, is the leading approximation** in that 1D run (the cutoff search lands a
factor of six below the momentum the lattice drives, and the renormalisation hides
the loss completely). Build it with `cd docs && pdflatex guide.tex && pdflatex guide.tex`.

## Install

```bash
pip install -e ../qutecipy --no-deps      # tensor cross interpolation + tensor trains
pip install numpy scipy jax               # jax is used only by the dense GP oracle
```

and put this directory on `sys.path` (the examples and tests do so themselves).

## Layout

| module | what |
|---|---|
| `grid.py` | `Grid`: the interleaved quantics grid in any dimension — index ↔ coordinate ↔ momentum |
| `pde.py` | `Equation`, `solve`, the split-step loop, function → train, function → diagonal MPO, the $d$-dimensional Fourier transform, `norm`/`normalise`/`integral`/`reconstruct`/`evaluate`/`to_dense` |
| `equations.py` | the presets and the Fermi–Dirac momentum window |
| `reference.py` | the dense solver running the same scheme |
| `tt.py` | tensor-train utilities: cores, δ-embedding into an MPO, MPO embedding into a subset of sites, overlaps, batched evaluation |
| `fourier.py` | the analytic quantics Fourier MPO (Chen–Lindsey construction) |
| `batcheval.py` | batched, memoised function evaluation for TCI, with a fast path for the one-point calls its pivot searches make |
| `fit.py` | variational MPO × MPS application, for operators whose rank makes the exact product too large |
| `operators.py`, `evolve1d.py`, `evolve2d.py`, `dense.py`, `observables.py` | the Gross–Pitaevskii solvers |

## Things worth knowing before trusting a number

- **TCI's reported error can be meaningless.** It is estimated on the pivots TCI itself
  chose and says nothing about a region no pivot reached — a function centred in the
  box sits on the most significant bit of every variable, and TCI started on one side
  can interpolate that side only while reporting convergence. The solver seeds both
  sides of the centre and the peaks of $|f|$ and runs a batched global pivot search
  after every sweep; still, check anything that matters with `pde.max_error` against
  an independent sample.
- **Rank is the cost and the accuracy.** `maxdim` caps the bond dimension of the field
  after every operation; `tolerance` sets the truncation. A field that develops fine
  structure (sharp fronts, spatiotemporal chaos) will hit the cap, and what happens
  then is a truncation, not a warning. Watch the ranks the solver logs.
- **A norm of 1.000000 proves nothing** once a momentum window or a renormalisation is
  in play: the window is not norm-preserving and the renormalisation hides whatever it
  discarded.
- **Fields are complex128** even for real equations. Take `.real` of what you
  reconstruct.
- **Indices are 64-bit**, so a grid has at most 62 sites: $R \le 31$ in 1D, 20 in 3D.
