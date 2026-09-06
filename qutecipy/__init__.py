"""qutecipy: quantics tensor cross interpolation in Python.

A port of TensorCrossInterpolation.jl (the TCI algorithm and its tensor-train
infrastructure) and QuanticsGrids.jl (coordinate <-> quantics-bitstring mapping),
plus two extensions with no Julia counterpart: array-valued TCI
(``crossinterpolate2_array``) and the layer that joins the two ports together
(``quantics_crossinterpolate``).

The ported API mirrors the Julia packages' exports (crossinterpolate1,
crossinterpolate2, TensorTrain, DiscretizedGrid, ...), adapted to 0-based indexing
-- see CLAUDE.md for the full porting plan and design decisions.
"""
from qutecipy.arrayvalued import (ArrayTensorTrain, ArrayValuedFunction,
                                  crossinterpolate2_array,
                                  estimate_componentweights)
from qutecipy.contraction import Contraction, contract
from qutecipy.conversion import tci1_from_tci2, tci2_from_tci1
from qutecipy.gausskronrod import kronrod
from qutecipy.integration import integrate
from qutecipy.quantics import (DiscretizedGrid, InherentDiscreteGrid,
                               quantics_function)
from qutecipy.quanticstci import QuanticsTensorCI, quantics_crossinterpolate
from qutecipy.tci1 import TensorCI1, crossinterpolate1
from qutecipy.tci2 import TensorCI2, crossinterpolate2, optimize
from qutecipy.tensortrain.base import AbstractTensorTrain
from qutecipy.tensortrain.cache import TTCache
from qutecipy.tensortrain.cachedfunction import CachedFunction
from qutecipy.tensortrain.core import TensorTrain, add, subtract, tensortrain
from qutecipy.util import optfirstpivot

__all__ = [
    "AbstractTensorTrain",
    "TensorTrain",
    "tensortrain",
    "add",
    "subtract",
    "TTCache",
    "CachedFunction",
    "TensorCI1",
    "crossinterpolate1",
    "TensorCI2",
    "crossinterpolate2",
    "optimize",
    "optfirstpivot",
    "tci1_from_tci2",
    "tci2_from_tci1",
    "kronrod",
    "integrate",
    "Contraction",
    "contract",
    "ArrayValuedFunction",
    "ArrayTensorTrain",
    "crossinterpolate2_array",
    "estimate_componentweights",
    "InherentDiscreteGrid",
    "DiscretizedGrid",
    "quantics_function",
    "QuanticsTensorCI",
    "quantics_crossinterpolate",
]
