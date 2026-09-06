"""Global numerical configuration.

Must be imported before any JAX array is created: ``jax_enable_x64`` cannot be
switched on after the fact, and without it every array is complex64, which makes
the ``tol=1e-10`` used throughout the original meaningless.
"""
import jax

jax.config.update("jax_enable_x64", True)

#: All wave functions and operators are complex128, matching Julia's ComplexF64.
CDTYPE = "complex128"
FDTYPE = "float64"
