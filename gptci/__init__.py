"""Backwards-compatible alias: the package is now called ``tnde``.

``import gptci`` and ``from gptci import evolve1d`` keep working, and resolve to the
*same* module objects as ``tnde`` (not second copies), so states and operators made
through either name are interchangeable.
"""
import importlib
import sys

import tnde

_SUBMODULES = ("config", "tt", "fourier", "operators", "batcheval", "fit", "dense",
               "evolve1d", "evolve2d", "observables", "grid", "pde", "equations",
               "reference")

for _name in _SUBMODULES:
    _mod = importlib.import_module(f"tnde.{_name}")
    sys.modules[f"gptci.{_name}"] = _mod
    globals()[_name] = _mod

__path__ = list(tnde.__path__)
del _name, _mod, importlib, sys
