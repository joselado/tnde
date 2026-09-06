"""tnde -- differential equations on tensor networks.

Time-dependent PDEs of the form ``d_t u = l(-i grad) u + p(x) u + N(u, x, t)`` are
evolved by Strang splitting on quantics tensor trains: the field on a ``2**R``-point
grid per direction is a train of ``R`` small tensors, every operator is a matrix
product operator, and the nonlinear step is re-interpolated by tensor cross
interpolation. See :mod:`tnde.pde` for the method and :mod:`tnde.equations` for the
ready-made equations.

The :mod:`tnde.evolve1d` / :mod:`tnde.evolve2d` drivers are the Gross-Pitaevskii
solvers the library grew out of; they reproduce a published quantics benchmark
tensor-for-tensor and keep that benchmark's conventions.
"""
from tnde import config  # noqa: F401  (must come first: enables x64)
from tnde import equations  # noqa: F401
from tnde.grid import Grid  # noqa: F401
from tnde.pde import (Equation, solve, interpolate, diagonal_mpo, multiplier_mpo,  # noqa: F401
                      fourier_transform, norm, normalise, integral, evaluate, to_dense,
                      reconstruct, max_error)
