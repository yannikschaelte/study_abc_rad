"""Conversion reaction problem, based on [#maierloo2017]_.

.. [#maierloo2017]
    Maier, Corinna, Carolin Loos, and Jan Hasenauer.
    "Robust parameter estimation for dynamical systems from outlier-corrupted
    data."
    Bioinformatics 33.5 (2017): 718-725.
"""

import numpy as np
from typing import Callable

from pyabc import Distribution, RV

from .base import Problem


class CRProblem(Problem):
    """Conversion reaction ODE-based problem."""

    def __init__(
        self,
        noise_std: float = 0.02,
    ):
        """
        :par noise_std: Noise standard deviation.
        """
        self.p_true = {"p0": -1.5, "p1": -1.5}
        self.limits = {"p0": (-3.5, 1), "p1": (-3.5, 1)}
        self.noise_std = noise_std
        self.n_t: int = 10
        self.t_max: int = 60
        self.ts = np.linspace(0, self.t_max, self.n_t)
        self.x0 = np.array([1.0, 0.0])

    def get_model(self) -> Callable:
        def model(p):
            # we assume that only y1 is measured
            y = x(p, self.x0, self.ts)[1, :]
            # add noise
            y += self.noise_std * np.random.normal(size=y.shape)
            return {"y": y.flatten()}

        return model

    def get_prior(self) -> Distribution:
        prior = Distribution(
            **{key: RV("uniform", lb, ub - lb) for key, (lb, ub) in self.limits.items()}
        )
        return prior

    def get_prior_bounds(self):
        return self.limits

    def get_obs(self) -> dict:
        return self.get_model()(self.p_true)

    def get_gt_par(self) -> dict:
        return self.p_true

    def get_id(self) -> str:
        return f"cr_{self.noise_std}"


def x(p, x0, ts):
    """
    States via analytic solution of ODE.
    Returns an array of shape n_x * n_t.
    """
    p0, p1 = 10 ** p["p0"], 10 ** p["p1"]
    n_t = len(ts)
    sol = np.zeros((2, n_t))
    for ix, t in enumerate(ts):
        e = np.exp(-(p0 + p1) * t)
        a = (
            1
            / (-p0 - p1)
            * np.array([[-p1 - p0 * e, -p1 + p1 * e], [-p0 + p0 * e, -p0 - p1 * e]])
        )
        sol[:, ix] = np.dot(a, x0).flatten()
    return sol
