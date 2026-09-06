# The physics of `gptci`

What equation is being solved, in what units, with what approximations, and what
each approximation costs. `PORTING_NOTES.md` covers the *engineering* — index
orders, conventions, benchmarks; this document covers what the numbers mean.

Everything quantitative below was measured against the code in this repository; the
measurement is named at each claim.

---

## 1. The equation

The Gross–Pitaevskii equation for a single condensate wave function, in units with
`hbar = 1`:

```
i d_t psi(x,t) = [ -(1/2m) d_x^2  +  V(x)  +  g |psi(x,t)|^2 ] psi(x,t)
```

with `psi` normalised to one, `int |psi|^2 dx = 1`. There is no separate particle
number: `N` is absorbed into `g`, which is the only interaction parameter. `g > 0`
is repulsive.

Nothing in the code carries dimensions. `m`, `g`, `V` and the box are pure numbers,
and the length and time units are whatever makes `hbar = 1` and the given `m` true.
For the paper's runs `m = 1`, so lengths are in oscillator units of a trap of unit
frequency and times in inverse units of the same.

Every sign convention is fixed in `dense.py`, which is the clearest statement of the
scheme in the repository:

| term | propagator | code |
|---|---|---|
| interaction | `exp(-i g |psi|^2 h)` | `dense.nonlinear_step` (`dense.py:106`) |
| potential | `exp(-i V(x) h)` | `dense.potential_phase` (`dense.py:115`) |
| kinetic | `exp(-i k^2 h / 2m)` | `dense.kinetic_factor` (`dense.py:75`) |

The interaction propagator is exact for a time step, not approximate: `|psi|^2` is
conserved by the term it generates, so `exp(-i g |psi|^2 h)` is the exact solution of
`i d_t psi = g|psi|^2 psi` over `h`. The splitting error comes entirely from the
kinetic term not commuting with the two real-space terms.

## 2. The scheme

Second-order Strang splitting, mixed spectral / real space. `K` is the kinetic
propagator (diagonal in momentum), `V` the potential phase and `N` the nonlinear
phase (both diagonal in position), `R` the renormalisation. Reading left to right in
time (`evolve1d.py:8`):

```
(V/2)(N/2) R  [ K R N V ]^nsteps  K R (N/2)(V/2) R          # 1D
(V/2)(N/2)    [ K R V N ]^nsteps  K R (V/2)(N/2)            # 2D
```

`V` and `N` are both diagonal in `x` and commute, so their order — swapped between
the 1D and 2D drivers — is cosmetic. The 1D and 2D orders differ only because the
Julia originals differ; they are kept apart deliberately.

Three consequences worth knowing before plotting anything:

**A run of `nsteps` covers `(nsteps+1)*dt`, not `nsteps*dt`.** There are `nsteps+1`
kinetic applications: one per loop iteration plus the closing one. The paper's 1D
run, `nsteps = 100` at `dt = 0.01`, evolves to `t = 1.01`.

**Snapshot `j` is not the state at `t = j*dt`.** It is taken mid-Trotter, at the end
of loop iteration `j`. At that point the kinetic propagator has been applied `j`
times but `V` and `N` have each been applied for `(j + 1/2)*dt`. So a snapshot is
half a step ahead in the real-space terms and cannot be compared with an exact
solution at `t = j*dt` better than `O(dt)`. This is the convention the reference
`.jld2` files use, so it is preserved.

If you need a properly symmetrised state at `t = j*dt`, run `evolve` with
`nsteps = j-1` and take the *returned final state*, not a snapshot. That path applies
`K` a total of `j` times (`j-1` in the loop plus the closing one) and accumulates
`dt/2 + (j-1)*dt + dt/2 = j*dt` of both `V` and `N` — symmetrised, at `t = j*dt`.

**Trotter error is `O(dt^2)` per step, `O(dt^2)` accumulated** for the symmetrised
end-to-end evolution — but the mid-Trotter snapshots are only `O(dt)` accurate for
the reason above.

## 3. Discretisation, and the dispersion relation

The box `[xmin, xmax]` of length `L` carries `M = 2^R` points. The 1D grid includes
the endpoint (`spacing L/(M-1)`), the 2D grid does not (`spacing L/M`); this
inconsistency is inherited and documented in `PORTING_NOTES.md`.

The kinetic step does **not** use the spectral dispersion `-k^2`. It uses the
eigenvalue of the three-point second difference at spacing `a = L/M`
(`operators.exp_lap_lowpass_mpo`, `dense.kinetic_factor`):

```
lambda_FD(k) = -(4 M^2 / L^2) sin^2(pi k / M),      k = 0 .. M-1
```

Two things follow, and both matter physically.

**The DFT index maps to physical momentum as `k_phys = 2 pi k / L`.** Expanding at
small `k`, `lambda_FD -> -(2 pi k / L)^2 = -k_phys^2`. Verified numerically for `k = 1..7` at
`R = 16`, `L = 1000`: the deviation is `3.8e-08`, which is the expected
`O((k_phys a)^2)` residual and nothing else. This map is independent
of `R` — it depends only on the box length — which is why a cutoff calibrated at one
resolution transfers to another.

**The dispersion saturates instead of growing.** `lambda_FD` is bounded below by
`-4/a^2` at the Nyquist edge, where the true `-k_phys^2` would be
`-(pi/a)^2 ≈ -9.87/a^2`. The error is the standard `O((k_phys a)^2)`
finite-difference one: `|lambda_FD|` is `8.1%` too small at `k_phys a = 1` and a factor
`2.5` too small at the Nyquist edge. High-momentum components therefore propagate too slowly. At the
paper's 1D parameters `a = 9.3e-7`, so this is far below every other error in the
calculation — but it is the reason the dense oracle and the tensor-train solver agree
to `1e-12` rather than only to the spectral answer, and it must not be "fixed" to
`-k^2` without changing both sides.

## 4. The momentum low-pass — the dominant approximation

Wrapped around the kinetic phase is a Fermi–Dirac window in DFT index space
(`operators.lowpass`):

```
W(k) = 1 + sigma(-(k - kcut) beta) - sigma(-(k - (kmax - kcut)) beta)
```

which is `1` at both *ends* of the index range — where the small physical momenta
live, since the DFT wraps negative momenta to high indices — and `0` across the
middle. It keeps `|k_phys| < 2 pi * kcut / L` and discards the rest.

**This is not physics.** It is there so the momentum-space operator has a small bond
dimension: a phase `exp(-i k^2 dt / 2m)` that oscillates all the way to the Nyquist
edge is not compressible, while one that is smoothly switched off is. The cost is
that the kinetic step is no longer unitary — it removes norm every time it is
applied — and the `normalise` call immediately after every kinetic application puts
that norm straight back.

> **Norm conservation is not an accuracy diagnostic in this code.** The norm is
> restored by construction at every step. A run that is discarding a percent of the
> wave function per hundred steps reports a norm of 1.000000 throughout.

### What the cutoff actually is, and what it costs

`kinetic_mpo` searches for `kcut`: it starts at `2^8`, applies the operator to a
fixed Gaussian trial state `(1/pi)^(1/4) exp(-x^2/2)`, and doubles until the trial
retains 99% of its norm. In 2D there is no search — `kcut = 2^8` is hardcoded, as in
the original.

The criterion is calibrated on that trial Gaussian, whose momentum spread is
`sigma_k = 1`. It knows nothing about the potential the run will actually use. For
the paper's 1D configuration the search therefore stops at:

| `kcut` | `k_phys` cutoff | trial norm retained |
|---|---|---|
| `2^8 = 256` | 1.61 | 0.9546 — fails |
| **`2^9 = 512`** | **3.22** | **0.99996 — accepted** |

(measured with `dense.kinetic_factor` at `R = 16`; confirmed against the cached
`R = 30` operator in `opcache/`, which retains `0.999989` of the TCI'd trial state.)

Every number in this section is reproduced by `python docs/check_cutoff.py`, which
runs in a few seconds.

Now compare that with the physics of the run. The 1D lattice is `5 sin^2(10 x)`, and
`sin^2(10x) = (1 - cos 20x)/2`, so the potential drives the wave function at
`k_phys = 20` — a factor of **6 above the cutoff**. The lattice imprints sidebands in
real space at every potential step, and the next kinetic step deletes them.

Measured with the dense oracle at `R = 18`, running the paper's exact 1D
configuration for 100 steps:

| `kcut` | `k_phys` cutoff | norm lost per kinetic step (max) | cumulative over 100 steps |
|---|---|---|---|
| `2^9` (what the search picks) | 3.22 | `3.2e-04` | **`3.2e-02`** |
| `2^12` | 25.7 | `1.6e-08` | `1.1e-06` |
| `2^14` | 102.9 | `0` | `-5e-14` |

and the effect on the observable the paper reports:

| | final width `sqrt(<x^2> - <x>^2)` at step 100 |
|---|---|
| `kcut = 2^9`, dense `R = 18` | 1.6686 |
| reference MPS, `R = 30` (the paper's own data) | 1.6421 |
| `kcut = 2^14` (cutoff converged), dense `R = 18` | **1.4684** |

`1 - overlap` between the `2^9` and `2^14` runs after 100 steps is `7.4e-03`.

So the tensor-train solver reproduces the reference to `1e-10` — and both are far from
the same scheme with the aperture opened: `+11.8%` in width for the reference against
the converged dense run, `+13.6%` comparing the two dense runs directly (the two
percentages differ only because the first pair also differs in `R` and in truncation). **The momentum cutoff,
not the bond dimension, is the leading approximation in the 1D configuration.** The
`maxdim = 14` truncation and the `tol = 1e-10` interpolation error are three to eight
orders of magnitude smaller.

The lattice sideband is visible in the paper's own committed data, suppressed but not
gone: Fourier-transforming the reference state at step 100 over `|x| < 20` puts
`1.5e-04` of the norm above `k_phys = 20`, peaked at `k_phys = -20.1`, against
`3.5e-14` in the initial state. The sideband is regenerated in real space and
re-filtered in momentum space every step, reaching a filtered quasi-steady amplitude
rather than its true one.

The 2D configuration is in the same regime, less severely. `L = 200` and
`kcut = 2^8` give a cutoff of `k_phys = 8.04`, against an initial momentum kick of
`k = 5`, a Gaussian spread of `1`, and a lattice at `k_phys = 2`. Measured with
`dense.evolve_2d`-equivalent stepping at `R = 11` over 20 steps: `7.2e-03` norm lost
cumulatively at `2^8`, `2.6e-08` at `2^9`, and `1 - overlap = 1.3e-02` between them.

**If you use this code for new physics, raise `kcut` until an observable stops
moving.** Pass it explicitly (`evolve(..., kcut=2**12)`), which also skips the search.
The operator's bond dimension grows slowly with it — the cached `R = 30` operator at
`2^9` has rank 17 — so this is affordable.

## 5. Why a tensor train works here

A quantics representation writes the grid index in binary and gives each bit its own
tensor: site `1` selects which half of the box, site `2` which quarter, down to site
`R` selecting between adjacent grid points. **A site is a length scale**, and the bond
dimension across a cut is the amount of correlation between scales coarser and finer
than that cut.

That is why this problem compresses. The 1D state is, to good approximation, a
smooth envelope of width `~1` multiplied by a lattice-periodic modulation of period
`pi/10 ≈ 0.31`. Those two structures live at well-separated scales and are nearly
factorised, so a cut between them carries little correlation. In a box of length
`1000` resolved to `2^30` points, the state is captured by 30 tensors of rank at most
14 — against `10^9` amplitudes stored densely. Bond dimension is the physical
diagnostic here: it is a measure of how far the state is from a scale-separated
product, and it is what grows when the dynamics genuinely creates structure.

Two consequences of taking that seriously:

- **The `2^30` grid is not resolution for its own sake.** It is what lets a box wide
  enough that the wave function never reaches the boundary (`|x| < 500`) coexist with
  a spacing fine enough to resolve the lattice (`a = 9.3e-7`, i.e. `3.4e5` points per
  lattice period). Neither is negotiable and densely they are incompatible.
- **The 2D train is interleaved**, `x1 y1 x2 y2 ... xR yR` (`batcheval.interleave`).
  A cut after site `2n` then separates *all* structure coarser than `L/2^n` from all
  structure finer, in both directions at once — the physically natural cut for an
  isotropic state. The alternative ordering, all `x` sites then all `y`, would put a
  cut between the two variables, and its bond dimension would be the full `x`–`y`
  entanglement of the state, which for the rotated lattice components
  `sin^2((x+y)/sqrt(2))` is large.

The nonlinearity is the one term that cannot be precomputed: `exp(-i g|psi|^2 dt)`
depends on the state, so it is re-interpolated from scratch at every step
(`evolve1d.apply_nonlinearity`). That is the dominant cost per step, and it is why
the interaction — not the trap or the lattice — is what makes this expensive.

## 6. The two configurations, as physics

### 1D — a breathing quench in a shallow optical lattice

`examples/reproduce_paper_1d.py`; `R = 30`, `x in [-500, 500]`, `dt = 0.01`,
`nsteps = 100`, `g = 5`, `m = 1`, `maxdim = 14`.

```
psi0(x) = (1/pi)^(1/4) exp(-x^2/2)
V(x)    = 0.01 x^2  +  5 sin^2(10 x)
```

- **The trap is not the one that made the state.** `psi0` is the ground state of a
  harmonic trap of frequency `1`; the trap it is released into is
  `V = 0.01 x^2 = (1/2) m omega^2 x^2` with `omega = sqrt(0.02) = 0.1414`, period
  `2 pi / omega = 44.4`. The trap's own ground-state width is
  `sqrt(1/(2 omega)) = 1.880` against the initial `sqrt(1/2) = 0.707`. The state is
  2.7x too narrow and expands: this is a breathing quench, which is what
  `tests/test_dense_breathing.py` gates the dense solver against analytically.
- **The interaction is strong and reinforces the expansion.** At the peak,
  `g |psi|^2 = 5 / sqrt(pi) = 2.82`, comparable to the lattice depth `5` and twenty
  times the trap frequency. The measured width goes `0.707 -> 1.046 -> 1.642` at steps
  1, 50, 100.
- **The lattice is shallow.** With `V = A sin^2(k_L x)`, `k_L = 10`, the recoil energy
  is `E_R = k_L^2 / 2m = 50`, so the depth `A = 5` is `0.1 E_R` — far too shallow to
  trap, and the sidebands it generates at `k_phys = +-2 k_L = +-20` are exactly what
  the low-pass of section 4 removes.
- **The run is short.** `t = 1.01` total, `2.3%` of a trap period. This is
  early-time expansion dynamics, not a breathing cycle.

### 2D — a moving wave packet in a four-fold quasi-periodic lattice

`examples/reproduce_paper_2d.py`; `R = 20` per direction (a `10^12`-point grid),
`x, y in [-100, 100]`, `dt = 0.01`, `nsteps = 20`, `g = 5`, `maxdim = 50`,
`kcut = 2^8` fixed.

```
psi0(x,y) = (1/pi)^(1/4) exp(-(x^2+y^2)/2) exp(5 i x)
V(x,y)    = 0.001 (x^2+y^2)
          + 10 [ sin^2(x) + sin^2(y) + sin^2((x+y)/sqrt(2)) + sin^2((y-x)/sqrt(2)) ]
```

- `exp(5ix)` is a **momentum kick**: the packet moves in `+x` at velocity `k/m = 5`,
  covering `~1` length unit over the run's `t = 0.21` — a fraction of a lattice period.
- The potential is **four sinusoidal modulations, two axis-aligned and two rotated by
  45 degrees**, with incommensurate periods (`pi` and `pi sqrt(2)`). This is a
  quasi-periodic potential, not a crystal: it has no unit cell, which is precisely the
  case where the interleaved quantics representation earns its keep, and it is the
  reason the state's rank grows fastest here.
- Depth `10` against `E_R = 1/2` for `k_L = 1` — this lattice is `20 E_R`, deep,
  unlike the 1D one.
- The trap `0.001 (x^2+y^2)` gives `omega = 0.0447`, period `141` — over the 20-step
  run it is essentially flat, present to keep the packet bound rather than to shape
  the dynamics.

## 7. Observables

`observables.py` provides position moments and reconstruction only:

| quantity | MPO route | direct-summation route |
|---|---|---|
| `<x>`, `<x^2>`, width | `width`, `expectation_value` | `width_dense` |
| 2D `<x> <y> <r> <r^2>`, widths | `moments_2d` | `moments_2d_dense` |
| `psi(x)`, `|psi(x,y)|^2` | — | `reconstruct`, `reconstruct_window`, `density_2d` |

**Prefer the `_dense` routes.** A quantics MPO for `x^2` is built by cross
interpolation to a tolerance *relative to* `max|x^2|`, which on the paper's box is
`2.5e5`. A tolerance of `1e-6` therefore buys an absolute accuracy of `~0.25` against
an expectation value of order `0.5`. This is why the original's committed
`widths.txt` begins at `0.8117` where the analytic answer is `sqrt(1/2) = 0.70711`;
`width_dense` returns `0.707107`. The `_dense` routes reconstruct the wave function on
a sample and sum directly, and are limited only by the sampling density — the state
is still never materialised on the full grid.

**Not implemented, and worth knowing:** there is no energy and no chemical potential.
The natural physical invariants of this system are the GP energy functional

```
E[psi] = int [ |d_x psi|^2 / 2m  +  V |psi|^2  +  (g/2) |psi|^4 ] dx
```

and `mu = E + (g/2) int |psi|^4 dx`. `E` is conserved by the exact dynamics and would
be the sharpest available check on the splitting — considerably sharper than the norm,
which as section 4 explains is restored by construction and diagnoses nothing. Both
are computable from the quantities the code already has (the kinetic term via the same
Fourier MPO, the rest as diagonal expectation values), but neither is written.

## 8. What each test pins, physically

| test | what it establishes |
|---|---|
| `test_dense_breathing.py` | The dense oracle reproduces the analytic breathing law `<x^2>(t) = <x^2>_0 cos^2(wt) + <p^2>_0 sin^2(wt)/(mw)^2` for a Gaussian in a harmonic trap. Pins the trap, the dispersion and the Trotter order against a closed-form solution. |
| `test_kinetic.py` | The tensor-train kinetic propagator matches the dense FFT. Pins the Fourier sign, the unitary normalisation, all three site reversals, the Fermi–Dirac window and the finite-difference dispersion at once. |
| `test_fourier.py` | The quantics Fourier MPO is the DFT. |
| `test_evolve1d.py`, `test_evolve2d.py` | Full evolution against the dense oracle, linear and nonlinear, at small `R`. |
| `test_reference_r30.py` | The `R = 30` run against the paper's own committed MPS tensors. |
| `test_tt.py`, `test_fit.py` | Tensor-train algebra and the variational MPO application. |

Note what this ladder does and does not establish. Every gate compares the tensor
train against a dense solver *using the same discretisation and the same low-pass*, or
against the original's own output. It shows the tensor-train machinery is faithful to
the scheme. It does not, anywhere, check the scheme against the continuum
Gross–Pitaevskii equation — except `test_dense_breathing.py`, which does, at `g = 0`
and `kcut = 2^12`.

## 9. Error budget

For the paper's 1D configuration, largest first:

| source | size | controlled by |
|---|---|---|
| momentum low-pass | `0.20` absolute in the width, `+13.6%` — measured (§4) | `kcut` |
| Trotter splitting | `O(dt^2) ~ 1e-4` — estimated, not measured | `dt` |
| MPS truncation | **not measured** — would need `maxdim = 14` against `maxdim >= 28` | `maxdim`, `tolerance` |
| cross-interpolation | `~1e-10` (`tolerance`), but see the caveat below | `tolerance` |
| finite-difference dispersion | `O((k_phys a)^2) ~ 1e-10` at `R = 30` | `R` |

The `7.7e-06` max pointwise difference quoted in the README is **not** a truncation
error: it is the disagreement between this port and the paper's own tensors, both
running at `maxdim = 14` with the same truncation, so it bounds porting fidelity and
says nothing about how much the bond-dimension cap costs.

Two traps in reading these:

- **TCI's reported error is estimated on its own pivots** and says nothing about a
  region no pivot reached. On the paper's initial state it has been observed to report
  `7e-11` while missing half the wave function. Use `tt.max_error` against an
  independent sample.
- **A converged norm means nothing** (§4).

## 10. Symbols

| symbol | code | meaning |
|---|---|---|
| `psi` | `psi`, `psi0` | wave function, `int|psi|^2 dx = 1` |
| `g` | `g` | interaction strength, `>0` repulsive |
| `m` | `m` | mass, `1` in both paper runs |
| `V` | `potentials` | external potential; a *list*, summed |
| `R` | `R` | quantics digits; grid is `2^R` points (per direction) |
| `M` | `1 << R` | grid points |
| `L` | `xmax - xmin` | box length |
| `a` | `L/M` or `L/(M-1)` | grid spacing (§3) |
| `k` | index `0..M-1` | DFT index |
| `k_phys` | — | `2 pi k / L` (§3) |
| `kcut` | `kcut` | low-pass in DFT index; physical cutoff `2 pi kcut / L` |
| `beta` | `beta` | sharpness of the Fermi–Dirac window, `2` |
| `dt` | `dt` | Trotter step |
| `chi` | `maxdim` | maximum bond dimension |

## References

- M. Niedermeier et al., *Solving the Gross–Pitaevskii equation with quantics tensor
  trains*, [arXiv:2507.04262](https://arxiv.org/abs/2507.04262); original Julia code at
  [MarcelNiedermeier/Gross-Pitaevskii-TCI](https://github.com/MarcelNiedermeier/Gross-Pitaevskii-TCI).
- J. Chen, M. Lindsey, *Direct interpolative construction of the discrete Fourier
  transform as a matrix product operator*,
  [arXiv:2404.03182](https://arxiv.org/abs/2404.03182) — the construction in `fourier.py`.
