# Vendored `pyitensor`

A verbatim copy of `dmrgpy`'s `pyitensor` package — a pure-Python implementation of
the subset of the ITensor v3 API that `dmrgpy` uses — vendored here for its **JAX
backend**, which is the device path for MPO × MPS. See "Running on a GPU" in the
top-level README for where it fits and what is still unmeasured.

- upstream: `git@github.com:joselado/dmrgpy.git`, path `src/dmrgpy/pyitensor`
  (local: `../dmrgpy`)
- commit:   `43d1a354c6abb80f896a11a3f9a50c2d2166682e`
- scope:    the `pyitensor/` package only.

It imports standalone: nothing in it reaches back into `dmrgpy` at import time. The
single reference outside the package — `from ..timedependent import sxt_to_skomega`
in `idmrg_window.py` — is a lazy import inside a function nothing here calls; every
other `mpscpp3` / `timedependent` mention is a comment. Verified by importing the
extracted tree with no `dmrgpy` on `sys.path`, and by switching the backend to JAX
and confirming `ITensor` data lands in `jaxlib` arrays.

There is no name collision with the dev machine's editable `dmrgpy` install: that
exposes the package as `dmrgpy.pyitensor`, not top-level `pyitensor`. Unlike
`qutecipy/`, this copy is what `import pyitensor` resolves to everywhere.

To refresh it against upstream:

```bash
rm -rf pyitensor
git -C ../dmrgpy archive HEAD src/dmrgpy/pyitensor | tar -x --strip-components=2 -C .
```

Do not edit these files here — change them upstream and re-vendor, or the two copies
drift.

## What `tnde` would use

`backend.py` (`set_backend`, `set_pad_bonds`, `set_jit`), `tensor.py`, `index.py`,
`svd.py`, `mpsalgebra.py` (`applyMPO`, `nmultMPO`, `_apply_chain`) and
`mpscontainer.py` (`MPS`, `MPO`). The rest — DMRG, TDVP, METTS, VUMPS, iDMRG — comes
along because it is one package and re-vendoring stays a clean copy that way; none of
it is on any path `tnde` takes.

`mpsalgebra.py::_apply_chain` is also the prior art `QUTECIPY_FINDINGS.md` cites: its
docstring documents the zip-up canonicalization hazard that `qutecipy.contract_zipup`
did not guard against.

`dmrgpy`'s own measurements for this backend live upstream in
`docs/pyitensor_gpu_port_plan.md` and `docs/gpu_cpu_performance.md`; they are for
DMRG / TDVP / METTS workloads, not this one.
