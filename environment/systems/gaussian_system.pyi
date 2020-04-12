from .abstract_system import AbstractSystem
import numpy as np

class GaussianSystem(AbstractSystem):
    _system: AbstractSystem
    _transition_noise_scale: float
    _measurement_noise_scale: float
    def __init__(
        self,
        system: AbstractSystem,
        transition_noise_scale: float,
        measurement_noise_scale: float = 0,
    ) -> None: ...
