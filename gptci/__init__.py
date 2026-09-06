"""Gross-Pitaevskii equation via quantics tensor cross interpolation, in Python/JAX.

A port of https://github.com/MarcelNiedermeier/Gross-Pitaevskii-TCI (arXiv:2507.04262),
built on ``qutecipy`` (TCI + quantics grids) and ``dmrgpy.pyitensor`` (MPS/MPO).
See PORTING_NOTES.md for the conventions, the reference data, and the gates.
"""
from gptci import config  # noqa: F401  (must come first: enables x64)
