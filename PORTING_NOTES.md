# Gross-Pitaevskii TCI → Python/JAX: plan and findings

Source: https://github.com/MarcelNiedermeier/Gross-Pitaevskii-TCI (paper: arXiv:2507.04262)
Cloned to `Gross-Pitaevskii-TCI/`.

## What is already solved, and by what

The cloned repo is ~2800 lines of *glue*; the algorithms live in its Julia
dependencies. With `qutecipy` and `dmrgpy.pyitensor` in hand, most of that stack
already exists in Python:

| Julia dependency | needed for | Python replacement | status |
|---|---|---|---|
| `TensorCrossInterpolation.jl` v0.9.14 | `crossinterpolate2`, `TensorTrain`, `contract(:naive)`, `reverse`, `CachedFunction`, `compress` | **`qutecipy`** (`github.com/joselado/qutecipy`) | ✅ complete, cross-validated against Julia |
| `QuanticsGrids.jl` v0.3.3 | `DiscretizedGrid`, coordinate ↔ quantics | **`qutecipy.quantics`** | ✅ complete |
| `ITensorMPS.jl` 0.3.6 | `MPS`/`MPO`, `contract(...; method="fit")` | not needed — see gap 2 | ✅ closed by measurement |
| `QuanticsTCI.jl` v0.7.0 | `quanticsfouriermpo` | **`gptci.fourier`** | ✅ ported (~40 lines) |
| `Quantics.jl` v0.4.5 | `fouriertransform` (2D path only) | **`gptci.evolve2d`** | ✅ ported via MPO embedding |

The pinned Julia sources are cloned to `reference_julia/` at exactly the versions
the paper uses — that is the spec to read when a convention is in question.

### Feasibility already demonstrated

`qutecipy.crossinterpolate2` reproduces the paper's R=30 initial state
**bond-dimension for bond-dimension** against the committed reference MPS, in 0.76 s:

```
linkdims (mine): [2,2,2,2,2,2,2,4,8,10,9,7,6,5,5,5,4,4,3,3,3,3,3,3,3,2,2,2,2]
linkdims (ref):  [2,2,2,2,2,2,2,4,8,10,9,7,6,5,5,5,4,4,3,3,3,3,3,3,3,2,2,2,2]
rank=10  err=4.9e-11
```

with pointwise agreement to ~1e-6 (the residual is only the reference's
normalisation, which was not applied here). So the hardest single component of
the port — TCI2 with prrLU — needs no work at all.

## Conventions — confirmed numerically, not inferred

Read out of `QuanticsGrids.jl` v0.3.3 and verified by evaluating the committed
`psi_mps_1.jld2`:

- **Julia is 1-based; qutecipy is 0-based** — in *both* the grid index and the
  quantics digits. Julia grid index `i` ↔ qutecipy `i-1`; Julia digit `{1,2}` ↔
  qutecipy `{0,1}`. Julia's explicit pivot `p1 = [2,1,1,…,1]` becomes `[1,0,0,…,0]`.
  This is the single most likely source of silent off-by-one damage. Confine it to
  one translation layer.
- **Digits are MSB-first.** `i = 1 + Σ_{n=1..R} (digit[R-n+1] - 1)·2^(n-1)` (Julia form).
- **`includeendpoint=True` ⇒ step `(xmax-xmin)/(2^R - 1)`**, `x_i = xmin + (i-1)·Δ`.
  qutecipy implements this by widening `upper_bound`; numerically identical.
- **But `fidelity_ITensor` / `normalise` use `dx = (xmax-xmin)/2^R`.** That
  inconsistency is in the original. Preserve it per-function; do not "fix" it, or
  the numbers stop matching.

Verification (step-1 reference MPS vs. analytic normalised Gaussian):

| x | \|MPS\| | `(1/π)^¼ e^{-x²/2}` | ratio |
|---|---|---|---|
| 0.000000 | 7.511255e-01 | 7.511255e-01 | 1.000000 |
| 0.931323 | 4.868186e-01 | 4.868186e-01 | 1.000000 |
| 4.656613 | 1.469202e-05 | 1.469202e-05 | 1.000000 |

## Reference data

`refdata/ref_mps_1D.npz` — the 11 committed `psi_mps_*.jld2` snapshots
(steps 1, 10, …, 100), converted by `refdata/dump_jld2.py`.
Keys `step{N}_core{i}` (complex128, Julia axis order `(left, phys, right)`), `step{N}_n`.

JLD2 is HDF5-based so `h5py` reads it, but HDF5 reports Julia's dimensions
*reversed*, so each core is transposed on load; complex is a `(re, im)` compound
type. The system `h5py` is ABI-broken against numpy 2.5.2 — the dump was done in a
throwaway venv, and the `.npz` is the artefact. No need to touch h5py again.

Bond dims: step 1 peaks at 10 (pure TCI, `tol=1e-10`); steps 10–100 sit flat at the
`maxdim=14` ceiling — i.e. **truncation-dominated**.

The identical bond profile across all ten snapshots is truncation saturation, not a
stationary state: the snapshots genuinely evolve. Normalised overlaps decay
monotonically and smoothly,

```
|<step10|step20>| = 0.9947   |<step10|step50>|  = 0.9372
|<step10|step30>| = 0.9804   |<step10|step100>| = 0.8229
```

so step-by-step matching against these tensors is a meaningful gate.

### Do NOT trust `Data_reconstructed/*.txt` as a tight target

- `widths.txt` row 1 is `0.8117`; the analytic width of the normalised unit Gaussian
  at t=0 is `sqrt(1/2) = 0.7071`. `observables_1D.jl` builds the x²-MPO at `tol=1e-6`
  *relative to* `max|x²| = 2.5e5` → absolute error ~0.25 against `⟨x²⟩ = 0.5`. The
  observable pipeline is tolerance-limited at O(0.1).
- The script `cd`s into `..._sine_potential2_test_run` but the committed directory is
  `..._sine_potential2test_run` (no underscore) — the `.txt` may come from a different
  run than the `.jld2`.
- `observables_1D.jl` uses `tol=1e-6, maxdim=10`; `GP_1D.jl` uses `tol=1e-10, maxdim=14`.
- Only ~100 of the configured 1000 steps were saved.

**Tight reference = the `.npz` tensors. Physics gate = analytic g=0 harmonic breathing.**

## The three gaps to fill

1. **`quanticsfouriermpo`** — the Chen–Lindsey interpolative DFT-as-MPO construction
   (`reference_julia/QuanticsTCI.jl/src/fouriertransform.jl`, ~40 lines of real code:
   a Chebyshev–Lagrange barycentric basis, one analytic core tensor, then
   `compress(:SVD, tolerance, maxbonddim=12)` and a `1/sqrt(2)` per-core normalisation).
   qutecipy already provides `TensorTrain.compress`, so this is a direct port.
2. **Variational fit MPO×MPS.** **Needed after all — but only in 2D.** Julia uses
   `ITensors.contract(mpo, mps; method="fit", nsweeps=2, maxdim)` for the *potential*
   steps and `TCI.contract(...; algorithm=:naive)` for the kinetic step. Applying a real
   `exp_potential_MPO` to the reference `step10` state at `maxdim=14`, measured against
   the untruncated exact product:

   | | `exp(-i·0.01x²·dt)` | `exp(-i·5sin²(10x)·dt)` |
   |---|---|---|
   | naive + global SVD | **7.4e-06** | **7.3e-06** |
   | zip-up | 5.1e-02 | 3.1e-01 |

   In **1D**, `contract(algorithm="naive")` is already at the optimal `maxdim=14`
   truncation, so it substitutes for the fit at ~1e-5 accuracy and is what `evolve1d`
   uses. The MPOs there have rank ≤ 17, so the exact intermediate is small.

   In **2D** that reverses. The momentum-space kinetic MPO has rank ~65 against a state
   at `maxdim=50`, so the exact intermediate carries bond dimensions near 3000 and a
   single application costs 45–240 s — the whole 2D bottleneck. `gptci/fit.py`
   implements the two-site variational sweep:

   | R=12, momentum MPO (rank 65) × state (rank 43) | time | agreement |
   |---|---|---|
   | `naive` + global SVD | 171.4 s | — |
   | variational fit, 2 sweeps | **1.4 s** | 2.2e-08 |

   a **123× speedup**, which takes the paper's 2D configuration from impractical to
   ~58 s per step at R=20.

   Two things had to be right for it to work at all, both silent when wrong:

   * **The sweep must re-gauge.** A two-site split that always leaves the orthogonality
     centre on the right breaks environment orthonormality on the way back; at 2D sizes
     the environments overflow to `inf` within one sweep and the SVD stops converging.
   * **`np.einsum(..., optimize=True)` picks a catastrophic path** for the four-operand
     environment updates — a single one did not finish in 300 s at 2D sizes, against
     ~30 ms once written as an explicit chain of `tensordot` calls. A factor of >10⁴.

   **How many sweeps?** Two, and more buys nothing. Pushing `maxdim` *below* the
   product's true rank and comparing against `naive`+SVD at the same `maxdim` (the
   near-optimal bond-limited answer), at R=6:

   | | maxdim 4 | maxdim 8 | maxdim 16 | maxdim 32 |
   |---|---|---|---|---|
   | momentum MPO (true rank 16) | 3.9e-02 | 2.6e-05 | 4.9e-14 | 4.9e-14 |
   | potential MPO (true rank 43) | 6.0e-02 | 3.2e-04 | 3.7e-06 | 5.4e-13 |

   Every entry is *identical* at `nsweeps` = 1, 2, 4 and 8. The sweeps converge
   immediately; the residual is the gap between the variational optimum and the global
   SVD optimum, and it shrinks with `maxdim`, not with sweep count. At the paper's 2D
   setting (`maxdim=50`, R=12) that gap is 2.2e-08.

   And the fit must **not** be used for the Fourier MPO: it is highly non-local, so two
   sweeps from a truncated-state guess converge poorly (2.8e-02 against `naive`'s
   7.5e-13), and it is cheap either way (1.6 s at R=20). The original makes the same
   split — `Quantics.fouriertransform` goes through ITensor's accurate `apply`, and only
   `apply_MPO_IT` uses `method="fit"`.

   **Separate finding — a real defect in `qutecipy.contract_zipup`.** It does not
   right-canonicalize its operands before the left-to-right sweep, so the singular values
   it truncates on are weighted by the norm of the untouched right part and the cutoff
   discards the wrong components. (This is exactly the hazard `pyitensor`'s
   `_apply_chain` docstring documents and guards against.) Right-canonicalizing both
   operands by hand first improves the sine case from `2.8e-01` to `4.8e-02` — a 6×
   improvement, confirming the cause. The residual gap to `naive`'s `7.3e-06` is what a
   single non-variational pass costs: zip-up truncates greedily as it sweeps and never
   revisits, so it does not reach the optimal bond-limited answer even when correctly
   gauged. The standard remedy is a variational refinement sweep — which is exactly what
   `gptci/fit.py` now does, reaching 5.4e-13 of the optimum where zip-up sits at 4.8e-02.
   Both halves are worth reporting upstream.
3. **`Quantics.fouriertransform`** (2D path only) — the interleaved-unfolding FT that
   `Fourier_transform_2D` / `kinetic_evolution_2D` use. Expressible via the same QFT
   MPO applied to one variable's sites; only needed once 2D is in scope.

Plus small glue: `TT_to_MPO` (δ-embedding of a TT into a diagonal MPO),
`IT_MPO_conversion` (TT → pyitensor `MPO`), and the 1-based↔0-based index layer.

## Where JAX goes — and where it must not

JAX earns its place at the *leaves*, not in the adaptive skeleton.

**JAX in:**
- `dense.py` — the split-step oracle, fully `jit`-able.
- **Batched function evaluation inside TCI.** qutecipy exposes a `BatchEvaluator`
  interface (`batchevaluate(leftindexset, rightindexset, ncent)`) that
  `crossinterpolate2` calls in batches. Subclass it with a `vmap`+`jit` kernel for
  `psi0`, `exp_pot`, and `lap_Fourier_lowpass`. Julia calls these scalar-at-a-time
  through `CachedFunction`; this is the single clearest JAX win.
- **`apply_f_tt`** — the nonlinear Trotter half-step TCIs a function that must
  evaluate the *current* MPS at scattered quantics points. Batched MPS evaluation is
  a `scan` of matmuls `vmap`ped over the batch. This is the hot loop of the solver.
- **MPO×MPS** via `pyitensor.backend.set_backend("jax")` +
  `set_pad_bonds(maxdim)` + `set_jit()`. pyitensor already implements exactly the
  pad-to-fixed-shape trick that stops `jax.jit` retracing on every bond-dimension
  change — and `maxdim=14` is fixed throughout this problem, so it fits perfectly.

**JAX out:** qutecipy's rrLU / pivot search / rank-adaptive sweeps. Dynamic shapes
and data-dependent branching; `jit` would recompile per rank and lose to NumPy.
Leave qutecipy untouched.

Non-negotiable: `jax.config.update("jax_enable_x64", True)` **before any array is
created**, or everything is complex64 and `tol=1e-10` is meaningless.

## Package layout

```
programs/
├── qutecipy/                 the TCI + quantics engine, pip install -e . --no-deps
│                             (do NOT symlink it into tnde/ -- a directory of that
│                              name there shadows the installed package)
└── tnde/
    ├── Gross-Pitaevskii-TCI/ cloned original (reference)
    ├── reference_julia/      pinned Julia deps (the spec)
    ├── refdata/              ref_mps_1D.npz + dump_jld2.py
    ├── tests/                the gates
    └── gptci/                <- the port
        ├── config.py         x64 enable  [done]
        ├── dense.py          JAX split-step oracle  [done]
        ├── tt.py             TT_to_MPO, inner, fidelity, normalise, batched eval  [done]
        ├── fourier.py        quanticsfouriermpo (gap 1)
        ├── operators.py      exp_potential_MPO, exp_lap_Fourier_MPO_lowpass,
        │                     get_kinetic_lowpass_mpo, pos/pos^2 MPOs
        ├── jaxeval.py        BatchEvaluator subclasses over vmapped jitted kernels
        ├── evolve1d.py       GP_Trotter_1D
        ├── observables.py    widths, heatmaps, expectation values
        └── evolve2d.py       2D (gap 3)
```

## Status: done, and validated against the paper's own data

Both dimensions are ported and gated. The strongest result: **the port reproduces the
paper's committed R = 30 reference MPS exactly**, at every saved snapshot.

Running `GP_1D.jl`'s exact parameters (2**30 grid points on [-500, 500], Gaussian in a
harmonic trap plus a fast sinusoidal lattice, g = 5, dt = 0.01, maxdim = 14) and
comparing against the tensors committed to the original repository:

| step | rank | ref rank | 1 - overlap | max pointwise rel. diff |
|---|---|---|---|---|
| 1 | 10 | 10 | 0.0 | 8.4e-11 |
| 10 | 14 | 14 | 3.0e-13 | 1.7e-07 |
| 50 | 14 | 14 | 9.8e-11 | 3.3e-06 |
| 100 | 14 | 14 | 2.2e-10 | 7.7e-06 |

at 3.03 s per Trotter step. Bond dimensions match the reference exactly throughout.

### The gates

| stage | gate | result |
|---|---|---|
| grid / quantics conventions | evaluate the reference MPS, compare to the analytic Gaussian | ratio 1.000000000 |
| TCI core | reproduce the reference linkdims | identical, 0.76 s |
| dense JAX oracle | harmonic breathing vs analytic; O(dx**2) convergence | 8.3e-05 at R=16, ratio 31.9 over 2 halvings |
| TT layer | reference round-trip, norm, elementwise-product MPO | 3.2e-13 |
| Fourier MPO | vs dense FFT, both signs, round trip | 8.4e-15 / 1.6e-14 / 6.2e-14 |
| kinetic operator | vs dense FFT at three cutoffs | 7e-13 - 1.3e-12 |
| 1D evolution | vs dense oracle, g = 0 and g = 5 | 2.5e-09 / 3.0e-09 |
| 1D at R=30 | vs the paper's reference MPS | 1 - overlap = 3.0e-13 at step 10 |
| 2D Fourier | vs dense fft2, transposition detected | 6.8e-14 |
| variational fit | vs exact product, diagonal and momentum MPOs | 1.0e-12 / 8.9e-13 |
| 2D evolution | vs dense oracle, g = 0 and g = 5 | 1.1e-12 / 1.1e-12 |

`./run_tests.sh` runs all of them.

## Two silent-failure bugs found along the way

**1. TCI can report convergence while being completely wrong.** Interpolating the
paper's own initial state -- a Gaussian on [-500, 500] at R = 30, with exactly the
single pivot `GP_1D.jl` uses -- TCI returned a rank-7 train with a *reported* error of
7e-11 and a *true* error of 1.0, in 1 run out of 6. The overlap with the correct state
was exactly 1/sqrt(2): it had interpolated one half of the Gaussian and left the other
half zero.

The cause is structural. TCI's error estimate is sampled on its own pivots, so it
cannot see a region no pivot ever reached; and the state is centred at `x = 0`, which
on this box sits exactly on the *most significant bit*, so the two halves live in
different top-level branches. Adding random pivots does not help (it makes it worse,
2 in 6) because greedy ascent walks every random start to the same peak.

The fix is to seed both straddling indices `2**(R-1)` and `2**(R-1) - 1`, which lie in
opposite MSB branches -- the same "get pivots in non-trivial branches" trick the Julia
code applies to its low-pass MPO but not to its wave function. With it, 0 runs in 6
fail. Raising `nsearchglobalpivot` from 5 to 20 also fixes it independently; the port
does both. `gptci.tt.max_error` exists to check any TCI result against an independent
sample rather than trusting the reported error.

**2. The same gauge bug, twice.** `qutecipy.contract_zipup` does not right-canonicalize its operands before the
left-to-right sweep, so it truncates on singular values weighted by the norm of the
untouched right part. Right-canonicalizing by hand first improves a representative
case from 2.8e-01 to 4.8e-02. The residual gap to `naive`'s 7.3e-06 is not diagnosed;
a single zip-up pass is suboptimal regardless. Reported for upstream; the port uses
`algorithm="naive"` throughout and is unaffected.

## Where JAX ended up -- and the measurement that put it there

The plan was to batch TCI's function evaluations through `vmap`, since Julia evaluates
them one scalar at a time. **Measured, that loses.** On this CPU:

| workload (R = 30) | Julia route | numpy | numpy + cache | jax |
|---|---|---|---|---|
| TCI of a Gaussian | 0.09 s | 0.12 s | -- | 1.25 s |
| one `apply_f_tt` step | 1.70 s | 2.57 s | **0.74 s** | 7.05 s |

TCI's batches are tiny -- 12 points for a plain function, 70 for `apply_f_tt` --
because the block size is set by the *rank*, which is 10-14 here. At ~1700 kernel calls
per Trotter step, JAX's per-dispatch floor dominates and neither padding nor jitting
recovers it. Bulk evaluation does not rescue it either: at 2**20 points in one call
JAX is still 2x slower than NumPy, because the cores are 14x14 and XLA's CPU backend
has nothing to exploit at that size.

What *does* pay is batching in NumPy **combined with memoising on the integer grid
index**: TCI requests the same point repeatedly (122805 requests, 19067 distinct -- a
6.4x redundancy), so deduplicating is worth more than the choice of array library.
That combination is **2.3x faster than the Julia route**.

So JAX's place in this port is:

* **`gptci.dense`** -- the split-step oracle, where the arrays are full 2**R FFTs and
  the whole time loop is one `jit`ed `fori_loop`. This is a genuine win and it is what
  makes the oracle cheap enough to gate every other stage against.
* **an option, not the default**, everywhere else: `gptci.batcheval` and
  `gptci.tt.evaluate` both take `backend="jax"`, worth revisiting on a GPU or at bond
  dimensions far above the 14 this problem uses.
* **never** inside qutecipy's adaptive TCI sweep, which has dynamic shapes and
  data-dependent branching.

## Traps that fail silently

- The 1-based↔0-based shift (see Conventions). One layer, tested.
- `TCI.reverse` around the Fourier MPOs encodes the QFT's inherent bit reversal.
  Wrong → silent garbage. The `dt=0` identity test in stage 4 is the check.
- The kinetic phase uses the **finite-difference** dispersion `-4M²/L² sin²(πk/M)`,
  not `-k²`. Keep it; the dense oracle must match it when comparing.
- `fidelity_ITensor` returns a *squared* overlap; `normalise` divides each of the R
  cores by `n^(1/R)` where `n = fidelity^(1/4)`. Self-consistent — don't "simplify".
- `get_kinetic_lowpass_mpo` contains a `while n < 0.99` loop that *searches* for a
  large enough `kcut`, starting at 8 and incrementing. It is not a fixed operator;
  its cost and result depend on `dt`, `m` and the trial Gaussian.
- 2D uses `unfoldingscheme=:interleaved` (digits ordered x₁y₁x₂y₂…) and routes the FT
  through `Quantics.fouriertransform` rather than `quanticsfouriermpo`.

## Environment

- Python 3.12.7 (anaconda), JAX 0.7.1 **CPU-only** (no CUDA jaxlib installed; an
  NVIDIA GPU is present but the driver is not responding — `nvidia-smi` fails).
- `dmrgpy` is an editable install pointing at `~/Documents/programs/dmrgpy/src/dmrgpy`;
  `pyitensor` imports, backend switchable (`numpy` default, `jax` available).
- `qutecipy` is at `../qutecipy`, installed with `pip install -e . --no-deps`
  (`--no-deps` so it cannot pull a numpy that breaks the anaconda env).
- Machine is shared with other jobs: pin compute with
  `MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 taskset -c <core> …`.
