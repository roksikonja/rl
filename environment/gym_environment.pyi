from .abstract_environment import AbstractEnvironment
import gym
from typing import Tuple
from .datatypes import State, Action

class GymEnvironment(AbstractEnvironment):
    env: gym.envs.registration
    _time: float
    def __init__(self, env_name: str, seed: int = None) -> None: ...
