from .abstract_system import AbstractSystem
from typing import Callable, Type
from scipy.integrate import OdeSolver, RK45

class ODESystem(AbstractSystem):
    step_size: float
    func: Callable
    integrator: Type[OdeSolver]
    def __init__(
        self,
        func: Callable,
        step_size: float,
        dim_state: int,
        dim_action: int,
        integrator: Type[OdeSolver] = RK45,
    ) -> None: ...
