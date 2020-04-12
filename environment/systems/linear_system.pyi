import numpy as np
from .abstract_system import AbstractSystem
from ..datatypes import State

class LinearSystem(AbstractSystem):
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    _state: State
