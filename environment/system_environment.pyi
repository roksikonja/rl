from .abstract_environment import AbstractEnvironment
from .datatypes import State, Action, Reward, Done
from typing import Tuple, Callable, Union
from .systems.abstract_system import AbstractSystem

class SystemEnvironment(AbstractEnvironment):
    reward: Callable[..., Reward]
    system: AbstractSystem
    termination: Callable[..., Done]
    initial_state: Callable[..., State]
    _time: float


    def __init__(self, system: AbstractSystem,
                 initial_state: Union[State, Callable[..., State]] = None,
                 reward: Callable[..., Reward] = None,
                 termination: Callable[..., Done] = None) -> None: ...
