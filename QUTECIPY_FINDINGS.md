# Two findings for `qutecipy`

Both surfaced while porting `Gross-Pitaevskii-TCI` against it (this directory). Both are
*silent* failures — the library reports success and returns a wrong answer. Reproducers
and measurements below; nothing here is speculative.

The port itself works around both, so nothing is blocked.

---

## 1. `contract_zipup` does not right-canonicalize its operands

`qutecipy/contraction.py::contract_zipup` sweeps left to right, factorizing at each cut,
without first bringing either operand into right-canonical form. The singular values it
truncates on are therefore weighted by the norm of the not-yet-contracted right part, so
the cutoff discards the wrong components. Zip-up's truncation is only meaningful when the
right side is orthonormal — that is the assumption in Stoudenmire & White,
*New J. Phys.* **12**, 055026 (2010), §3.2.

**Measured.** `exp(-i·5·sin²(10x)·dt)` as a diagonal MPO applied to the paper's R=30,
rank-14 reference state at `maxbonddim=14`, against the untruncated exact product:

| | relative error |
|---|---|
| `algorithm="naive"` (exact product + global SVD) | **7.3e-06** |
| `algorithm="zipup"` | 2.8e-01 |
| `algorithm="zipup"`, operands right-canonicalized by hand first | 4.8e-02 |

The 6× improvement from hand-canonicalizing confirms the cause. The residual is what a
single non-variational pass costs: zip-up truncates greedily as it sweeps and never
revisits, so it does not reach the optimal bond-limited answer even when correctly
gauged. A variational refinement sweep is the standard remedy — `tnde/fit.py` in this
directory implements one and reaches 5.4e-13 of the optimal truncation.

**Prior art in the same codebase family.** `dmrgpy/src/dmrgpy/pyitensor/mpsalgebra.py::_apply_chain`
documents this exact hazard at length and calls `chain.position(1)` on both operands
before sweeping. Its comment records the symptom it caused there: ⟨Hb|Hb⟩ and ⟨b|H²b⟩
disagreeing by 0.86% where compiled ITensor agrees to 1e-15.

---

## 2. `crossinterpolate2` can report convergence while being completely wrong

When the function's support straddles a **high-order bit boundary**, TCI can interpolate
one branch and leave the other zero. Its error estimate is sampled on its own pivots, so
it never sees the missing half and reports convergence.

**Measured.** `(1/π)^(1/4)·exp(-x²/2)` on `[-500, 500]` at R=30, `tolerance=1e-10`, with a
single initial pivot at index `2**29` (which is what `GP_1D.jl` uses — the state is
centred at `x = 0`, i.e. exactly on the most significant bit):

| strategy | silently wrong | note |
|---|---|---|
| single centre pivot | **1 / 6 runs** | rank 7, reported error 7e-11, **true error 1.0** |
| centre pivot + 5 random, each `optfirstpivot`-refined | 2 / 6 runs | *worse* |
| both straddling indices `2**(R-1)` and `2**(R-1) - 1` | 0 / 6 runs | opposite MSB branches |
| single centre pivot, `nsearchglobalpivot=20` | 0 / 6 runs | |

On the failing runs the overlap with the correct state is exactly `1/sqrt(2)` — half the
probability, i.e. one side of the Gaussian.

Random pivots make it *worse* because `optfirstpivot` is a greedy coordinate ascent on
`|f|`, so every random start walks to the same peak and adds nothing new.

**Suggestions**, in rough order of cost:

- a docs note that the reported error is a pivot-sampled estimate and is not a bound;
- raise the default `nsearchglobalpivot` (5 → 20 fixed it here);
- when seeding from found peaks, also seed each peak's neighbours `i-1` and `i+1` — they
  lie in opposite branches whenever `i` sits on a bit boundary, which costs nothing and
  is what the Julia code already does by hand for its low-pass MPO
  (`pivots = [ones, 2*ones]`, commented "get pivots in non-trivial branches").

---

## Reproducers in this directory

- `refdata/ref_mps_1D.npz` — the paper's own reference MPS tensors, for a ground truth.
- `tnde/tt.py::max_error` — checks a tensor train against an independent sample of the
  function it is meant to represent, rather than trusting the reported error.
- `tnde/fit.py` — two-site variational MPO × MPS fit. Also worth a look for two bugs of
  its own that took a while to find: a two-site split that always leaves the
  orthogonality centre on the right destroys environment orthonormality on the return
  sweep (environments overflow to `inf`), and `np.einsum(..., optimize=True)` picks a
  catastrophic contraction path for the four-operand environment updates — >300 s versus
  ~30 ms for the same contraction written as explicit `tensordot` calls.
