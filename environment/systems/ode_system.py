"""Implementation of a Class that constructs an dynamic system from an ode."""

from .abstract_system import AbstractSystem
from scipy import integrate
import numpy as np


class ODESystem(AbstractSystem):
    """A class that constructs an dynamic system from an ode.

    Parameters
    ----------
    func : callable.
        func is the right hand side of as xdot = f(t, x).
        with actions, extend x to be states, actions.
    step_size : float
    dim_state: int
    dim_action : int
    # integrator : integrate.OdeSolver
    """

    def __init__(
        self, func, step_size, dim_state, dim_action, integrator=integrate.RK45
    ):
        super().__init__(
            dim_state=dim_state, dim_action=dim_action,
        )

        self.step_size = step_size
        self.func = func
        self._state = np.zeros(dim_state)
        self._time = 0
        self.integrator = integrator

    def step(self, action):
        """See `AbstractSystem.step'."""
        integrator = self.integrator(
            lambda t, y: self.func(t, y, action), 0, self.state, t_bound=self.step_size
        )

        while integrator.status == "running":
            integrator.step()
        self.state = integrator.y
        self._time += self.step_size

        return self.state

    def reset(self, state=None):
        """See `AbstractSystem.reset'."""
        self.state = state
        return self.state

    @property
    def state(self):
        """See `AbstractSystem.state'."""
        return self._state

    @state.setter
    def state(self, value):
        self._state = value

    @property
    def time(self):
        """See `AbstractSystem.time'."""
        return self._time
