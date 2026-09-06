# Vendored `qutecipy`

This directory is a verbatim copy of the `qutecipy` package, vendored so that a
plain clone of `tnde` runs with no external install beyond numpy/scipy/jax.

- upstream: `https://github.com/joselado/qutecipy` (local: `../qutecipy`)
- commit:   `685e2d7a2a3c702cda31bfb7701165345ca2b8d7`
- license:  MIT
- scope:    the `qutecipy/` package only; upstream's own tests, examples and
            `pyproject.toml` are not copied.

It is picked up by the `sys.path.insert(0, ROOT)` that every test and example
does, so nothing needs to be installed. To refresh it against upstream:

```bash
rm -rf qutecipy
git -C ../qutecipy archive HEAD qutecipy | tar -x -C .
```

Do not edit these files here — change them upstream and re-vendor, or the two
copies drift.
