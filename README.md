# gptci — Gross-Pitaevskii via quantics tensor cross interpolation, in Python

A Python port of [Gross-Pitaevskii-TCI](https://github.com/MarcelNiedermeier/Gross-Pitaevskii-TCI)
([arXiv:2507.04262](https://arxiv.org/abs/2507.04262)): the Gross-Pitaevskii equation
solved in 1D and 2D by a mixed-spectral second-order Trotter scheme on matrix product
states, with the wave function represented as a quantics tensor train.

The point of the method is that a 2^30-point grid never exists in memory. The state is
a train of 30 rank-≤14 tensors, and every operator — the potential, the propagator, the
`g|ψ|²` nonlinearity — is built by cross interpolation from a few thousand function
evaluations.

## It reproduces the paper's own data

Running `GP_1D.jl`'s exact parameters (2³⁰ points on [-500, 500], Gaussian in a
harmonic trap plus a fast sinusoidal lattice, g = 5, dt = 0.01, maxdim = 14) and
comparing against the MPS tensors committed to the original repository:

| step | rank | ref rank | 1 − overlap | max pointwise rel. diff |
|---|---|---|---|---|
| 1 | 10 | 10 | 0.0 | 8.4e-11 |
| 10 | 14 | 14 | 3.0e-13 | 1.7e-07 |
| 50 | 14 | 14 | 9.8e-11 | 3.3e-06 |
| 100 | 14 | 14 | 2.2e-10 | 7.7e-06 |

at 3.03 s per Trotter step, bond dimensions matching exactly throughout.

## Install

```bash
pip install -e ../qutecipy --no-deps      # TCI + quantics grids
pip install jax scipy numpy               # jax is used by the dense oracle
```

### Reference sources

The Julia original and its dependencies are cloned for reference but not vendored here
(each carries its own `.git`). To restore them:

```bash
git clone https://github.com/MarcelNiedermeier/Gross-Pitaevskii-TCI.git
mkdir -p reference_julia && cd reference_julia
for r in TensorCrossInterpolation.jl:v0.9.14 QuanticsGrids.jl:v0.3.3 \
         QuanticsTCI.jl:v0.7.0 Quantics.jl:v0.4.5; do
  n=${r%%:*}; v=${r##*:}
  git clone --depth 1 https://github.com/tensor4all/$n.git
  git -C $n fetch -q --depth 1 origin tag $v && git -C $n checkout -q $v
done
```

Those are the exact versions the paper pins, and they are the spec whenever a
convention is in question.

## Use

```python
import numpy as np
from gptci import evolve1d, observables

psi0 = lambda x: (1/np.pi)**0.25 * np.exp(-x**2/2)
pots = [lambda x: 0.01*x**2, lambda x: 5.0*np.sin(10*x)**2]

psi, snaps = evolve1d.evolve(psi0, pots, R=30, xmin=-500., xmax=500.,
                             g=5.0, dt=0.01, nsteps=100, maxdim=14)

x, values = observables.reconstruct_window(psi, 30, -500., 500., x0=0., halfwidth=20.)
width      = observables.width_dense(psi, 30, -500., 500., halfwidth=50.)
```

2D is `gptci.evolve2d.evolve`, with the same shape of call plus `ymin`/`ymax`.

Worked examples: `examples/reproduce_paper_1d.py`, `examples/reproduce_paper_2d.py`.

## Layout

| module | what |
|---|---|
| `dense.py` | dense split-step solver in JAX — the oracle every other stage is gated against (R ≲ 20) |
| `tt.py` | tensor-train utilities: δ-embedding into an MPO, overlaps, normalisation, batched evaluation |
| `fourier.py` | the quantics Fourier MPO (Chen–Lindsey), ported from `QuanticsTCI.jl` |
| `operators.py` | potential and position MPOs, the low-pass, the kinetic propagator and its cutoff search |
| `batcheval.py` | batched + memoised function evaluation for TCI, with a switchable array backend |
| `evolve1d.py` | the 1D Trotter driver |
| `fit.py` | variational (sweeping) MPO×MPS application — 123× faster than forming the exact product when the MPO's rank is large, which is what makes 2D practical |
| `evolve2d.py` | the 2D driver: interleaved unfolding, embedded Fourier MPOs |
| `observables.py` | widths, expectation values (1D and 2D moments), reconstruction, heatmaps |

## 2D

The 2D path stores an *interleaved* train over `x₁y₁x₂y₂…`, so the Fourier transform is
embedded into every other site. It is validated against a dense 2D FFT rather than
reference data (the original repository ships none for 2D): kinetic step and full
evolution both agree to ~1e-12, linear and nonlinear.

The paper's 2D configuration — R = 20, i.e. a 2²⁰ × 2²⁰ = 10¹² point grid, a moving
Gaussian in a four-fold quasi-periodic lattice — runs at 73 s per Trotter step (20 steps in 24 min).

One convention worth knowing: the 2D transform returns a train ordered over `(ky, kx)`,
*not* `(kx, ky)`. Each variable's QFT reverses its own digits in place and the final
whole-train reverse swaps which slot belongs to which variable. The original gets away
with it because its momentum operator is symmetric under `kx ↔ ky`; an asymmetric one
would be silently transposed.

## Tests

`./run_tests.sh` — the gates, each pinning one layer against an independent reference
(analytic solutions, a dense FFT, or the paper's own tensors). They take a few minutes.

## Notes

`docs/guide.tex` is the physics user guide: the equation and its units, the splitting
and what a "step" means on a time axis, the finite-difference dispersion and the
DFT-index-to-momentum map, why a quantics train compresses this problem, the two paper
configurations read as physics, and an error budget. Build it with
`cd docs && pdflatex guide.tex && pdflatex guide.tex` (twice, for the table of
contents). Its main finding is that the
**momentum low-pass, not the bond dimension, is the leading approximation** in the 1D
run — the cutoff search calibrates on a Gaussian trial state and lands a factor of six
below the momentum the lattice drives, and the renormalisation after every kinetic step
hides the resulting loss completely.

`PORTING_NOTES.md` records the conventions (several of which are inconsistent in the
original and are preserved deliberately), two silent-failure bugs found along the way,
and the benchmark that decided where JAX does and does not belong here.
`QUTECIPY_FINDINGS.md` is a standalone write-up of the two `qutecipy` bugs, with
reproducers, for reporting upstream.

Three things worth knowing before trusting a number out of this code:

- **TCI's reported error can be meaningless.** It is estimated on the pivots TCI itself
  chose, so it says nothing about a region no pivot reached. On the paper's own initial
  state it reports 7e-11 while missing half the wave function. Use `tt.max_error` to
  check against an independent sample.
- **A converged norm means nothing.** The kinetic step's low-pass filter is not
  norm-preserving, and the state is renormalised immediately after every application,
  so the reported norm is 1.000000 whatever the filter discarded. See `docs/guide.tex`.
- **Position-operator observables are tolerance-limited.** A quantics MPO for `x²` is
  built to a tolerance *relative to* `max|x²|`, which on a box of half-width 500 is
  2.5e5 — so `tol=1e-6` buys an absolute accuracy of ~0.25 against an expectation value
  of order 0.5. This is why the original's committed `widths.txt` starts at 0.8117 where
  the analytic answer is `sqrt(1/2) = 0.7071`. `observables.width_dense` avoids it and
  returns 0.707106781.
