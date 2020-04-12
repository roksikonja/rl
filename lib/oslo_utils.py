import logging
import warnings
from abc import ABCMeta, abstractmethod
from typing import Union

import cvxpy as cp
import numpy as np


class AbstractController(object, metaclass=ABCMeta):
    def __init__(self, dim_state, dim_action, initial_policy=None, initial_state=None):
        super().__init__()
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.dim_state_action = dim_state + dim_action
        if initial_policy is None:
            initial_policy = lambda: np.zeros(shape=(dim_action, dim_state))
        if initial_state is None:
            initial_state = lambda: np.zeros(shape=(dim_state, 1))
        self.initial_policy = initial_policy()
        self.policy = initial_policy()
        self.initial_state = initial_state()
        self.state = initial_state()
        self._time = 0

    @property  # type: ignore
    @abstractmethod
    def get_policy(self):
        raise NotImplementedError

    @abstractmethod
    def set_policy(self, value):
        raise NotImplementedError

    @property
    def time(self):
        return self._time


class CohenController(AbstractController):
    def __init__(
        self,
        Q: np.ndarray,
        R: np.ndarray,
        system,
        sigma: float,
        theta: float,
        nu: float,
        delta: float = 0.1,
        time_horizon: int = 1000,
        A0=None,
        B0=None,
        W=None,
        alpha_0=None,
        initial_state=None,
        initial_policy=None,
    ):
        self.Q = Q
        self.R = R
        self.system = system
        self.sigma = sigma

        if alpha_0 is None:
            _, S_Q, _ = np.linalg.svd(np.atleast_2d(Q))
            _, S_R, _ = np.linalg.svd(np.atleast_2d(R))
            alpha_0 = min(S_Q[-1], S_R[-1])
        self.alpha_0 = alpha_0
        self.theta = theta
        dim_state, dim_action = self.Q.shape[0], self.R.shape[0]
        self.nu = nu
        self.delta = delta
        self.time_horizon = time_horizon

        super().__init__(
            dim_state=dim_state,
            dim_action=dim_action,
            initial_policy=initial_policy,
            initial_state=initial_state,
        )
        self.S = np.zeros(shape=(self.dim_state_action, self.dim_state_action))
        if A0 is None:
            A0 = np.zeros(shape=(self.dim_state, self.dim_state))
        if B0 is None:
            B0 = np.zeros(shape=(self.dim_state, self.dim_action))
        self.A0 = A0
        self.B0 = B0

        self.A: np.ndarray = A0
        self.B: np.ndarray = B0

        if W is None:
            W = sigma ** 2 * np.identity(self.dim_state)
        self.W = W

        # The joint parameter space of A and B
        self.AB0: np.ndarray = np.block([self.A0, self.B0])

        # FIRST SET OF THE PARAMETERS
        # self.mu: float = 5 * self.theta * np.sqrt(self.time_horizon)
        # self.lam: float = 2 ** 11 * self.nu ** 5 * self.theta * np.sqrt(self.time_horizon) / (
        #         self.alpha_0 ** 5 * self.sigma ** 10)
        # self.beta: float = 2 ** 18 * self.nu * 4 * self.dim_state_action ** 2 / (
        #         self.alpha_0 ** 4 * self.sigma ** 6) * np.log(self.time_horizon / self.delta)

        # SECOND SET OF THE PARAMETERS
        # self.mu: float = 5 * self.theta * np.sqrt(self.time_horizon)
        # self.lam: float = self.nu ** 5 * self.theta * np.sqrt(self.time_horizon) / (
        #         self.alpha_0 ** 5 * self.sigma ** 10)
        # self.beta: float = self.nu * 4 * self.dim_state_action ** 2 / (
        #         self.alpha_0 ** 4 * self.sigma ** 6) * np.log(self.time_horizon / self.delta)

        # THIRD SET OF THE PARAMETERS
        self.mu: float = 1
        self.lam: float = 1
        self.beta: float = 1

        self.V0: np.ndarray = self.lam * np.identity(self.dim_state_action)
        self.V1: np.ndarray = self.lam * np.identity(self.dim_state_action)

        self.action: np.ndarray = self._propose_action()

        self.Z_top_Z = np.zeros(shape=(self.dim_state_action, self.dim_state_action))
        self.X_top_Z = np.zeros(shape=(self.dim_state, self.dim_state_action))

        self._z = np.concatenate((self.state, self.action))
        self._x_history = np.zeros(shape=(self.time_horizon, self.dim_state))
        self._z_history = np.zeros(shape=(self.time_horizon, self.dim_state_action))

        self._X_top_Z = np.zeros(shape=(self.dim_state, self.dim_state_action))
        self._Z_top_Z = np.zeros(shape=(self.dim_state_action, self.dim_state_action))
        self.control_update_counter = 0

    def _propose_action(self) -> np.ndarray:
        return self.policy @ self.state

    def step(self, observation: np.ndarray) -> Union[np.ndarray, None]:
        if self._time >= self.time_horizon:
            warnings.warn(
                "You have reached time horizon, the algorithm provides no more actions!"
            )
            return None
        self._x_history = np.roll(self._x_history, 1, axis=0)
        self._x_history[0, :] = observation.T
        self.state = observation
        self._X_top_Z = self._X_top_Z + self.state.reshape(-1, 1) @ self._z.reshape(
            1, -1
        )
        # print(self._X_top_Z)
        if (
            self._time == 0
            or np.linalg.det(self.V1) > 2 * np.linalg.det(self.V0)
            or self._time % 500 == 0
        ):
            logging.info(
                "Controller update number: ".format(self.control_update_counter)
            )
            # print('Controller update number: ', self.control_update_counter)
            # print('Current time: ', self._time)
            self.control_update_counter += 1
            self.V0 = self.V1
            # Cohen proposal
            AB = (1 / self.beta * self._X_top_Z + self.lam * self.AB0) @ np.linalg.inv(
                1 / self.beta * self._Z_top_Z
                + self.lam * np.identity(self.dim_state_action)
            )

            # Ridge regression
            # AB = (1 / self.beta * self._X_top_Z) @ np.linalg.inv(
            #     1 / self.beta * self._Z_top_Z + self.lam * np.identity(self.dim_state_action))

            # No punishment for A, B being large
            # AB = self._X_top_Z @ np.linalg.pinv(self._Z_top_Z)

            self.A: np.ndarray = AB[:, : self.dim_state]
            self.B: np.ndarray = AB[:, self.dim_state :]
            S = self._relaxed_sdp(
                self.A, self.B, self.Q, self.R, self.W, self.mu, self.V1
            )
            self.policy = S[self.dim_state :, : self.dim_state] @ np.linalg.inv(
                S[: self.dim_state, : self.dim_state]
            )
        self.action = self._propose_action()
        self._z = np.concatenate((self.state, self.action))
        self._Z_top_Z = self._Z_top_Z + self._z.reshape(-1, 1) @ self._z.reshape(1, -1)
        self._z_history = np.roll(self._z_history, 1, axis=0)
        self._z_history[0, :] = self._z.T
        self.V1 = self.V1 + 1 / self.beta * self._z @ self._z.T
        self._time += 1
        return self.action

    def _relaxed_sdp(
        self,
        A: np.ndarray,
        B: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        W: np.ndarray,
        mu: float,
        V: np.ndarray,
    ) -> np.ndarray:
        d, k = self.dim_state, self.dim_action
        n = self.dim_state_action
        q_r_block = np.block([[Q, np.zeros(shape=(d, k))], [np.zeros(shape=(k, d)), R]])
        S = cp.Variable((n, n), symmetric=True)
        constraints = [S >> 0]
        constraints += [
            S[:d, :d]
            >> np.block([A, B]) @ S @ np.block([A, B]).T
            + W
            - mu * cp.trace(S.T @ np.linalg.inv(V)) * np.identity(d)
        ]
        problem = cp.Problem(cp.Minimize(cp.trace(S.T @ q_r_block)), constraints)

        result = problem.solve(solver="MOSEK")
        # result = problem.solve(solver="CVXOPT")
        print(result)
        if problem.status == "infeasible":
            print(A, B, Q, R, W, mu, V)
        self.S = S.value
        return S.value

    def get_policy(self) -> np.ndarray:
        return self.policy

    def set_policy(self, policy: np.ndarray):
        self.policy = policy

    def time(self) -> int:
        return self._time
