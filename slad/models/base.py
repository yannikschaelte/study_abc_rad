"""Base problem definition."""

from typing import Any, Callable, Dict, List
import numpy as np
from abc import ABC, abstractmethod
import pyabc
from pyabc import Distribution, RV


class Problem(ABC):
    """Base problem class."""

    @abstractmethod
    def get_model(self) -> Callable:
        """Get the model."""

    @abstractmethod
    def get_prior(self) -> pyabc.Distribution:
        """Get the prior."""

    @abstractmethod
    def get_prior_bounds(self) -> dict:
        """Get prior boundaries."""

    def get_viz_bounds(self) -> dict:
        """Get boundaries for visualization."""
        return self.get_prior_bounds()

    def get_obs(self) -> dict:
        """Get the observation."""
        return self.get_model()(self.get_gt_par())

    @abstractmethod
    def get_gt_par(self) -> dict:
        """Get the ground truth parameter values."""

    def get_sumstat(self) -> pyabc.Sumstat:
        """Get the summary statistic function."""
        return pyabc.IdentitySumstat()

    def get_ana_args(self) -> Dict[str, Any]:
        """Get analysis arguments."""
        return {"population_size": 1000, "max_total_nr_simulations": 100000}

    @abstractmethod
    def get_id(self) -> str:
        """Get a problem identifier."""

    def get_y_keys(self) -> List[str]:
        """Get data keys."""
        return list(self.get_model()(self.get_gt_par()).keys())


class CoreProblem(Problem):
    """Core problem illustrating various challenging problem features."""

    def __init__(
        self,
        n0: int = 1,
        n1: int = 1,
        n2: int = 1,
        n3: int = 1,
        n4: int = 1,
        std0: float = 1e-2,
        std1: float = 1e0,
        std2: float = 1e2,
        std3: float = 1e-4,
        std4: float = 1e-4,
        ub0: float = 1e-1,
        ub1: float = 1e3,
        ub2: float = 1e-1,
    ):
        self.n0, self.n1, self.n2, self.n3, self.n4 = n0, n1, n2, n3, n4
        self.std0, self.std1, self.std2, self.std3, self.std4 = (
            std0,
            std1,
            std2,
            std3,
            std4,
        )
        self.ub0, self.ub1, self.ub2 = ub0, ub1, ub2

    def get_model(self):
        def model(p):
            return {
                "s0": p["p0"]
                + self.std0 * np.sqrt(self.n0) * np.random.normal(size=self.n0),
                "s1": p["p1"]
                + self.std1 * np.sqrt(self.n1) * np.random.normal(size=self.n1),
                "s2": 500
                + self.std2 * np.sqrt(self.n2) * np.random.normal(size=self.n2),
                "s3": 0.5
                + self.std3 * np.sqrt(self.n3) * np.random.normal(size=self.n3),
                "s4": p["p2"]
                + self.std4 * np.sqrt(self.n4) * np.random.normal(size=self.n4),
            }

        return model

    def get_prior(self) -> dict:
        return Distribution(
            p0=RV("uniform", 0, self.ub0),
            p1=RV("uniform", 0, self.ub1),
            p2=RV("uniform", 0, self.ub2),
        )

    def get_prior_bounds(self) -> dict:
        gt_par = self.get_gt_par()
        s = 10
        return {
            "p0": (
                max(gt_par["p0"] - s * self.std0, 0),
                min(gt_par["p0"] + s * self.std0, self.ub0),
            ),
            "p1": (
                max(gt_par["p1"] - s * self.std1, 0),
                min(gt_par["p1"] + s * self.std1, self.ub1),
            ),
            "p2": (0, self.ub2),
        }
        # return {'p0': (0, self.ub0), 'p1': (0, self.ub1)}

    def get_gt_par(self) -> dict:
        return {
            "p0": 0.4 * self.ub0,
            "p1": 0.4 * self.ub1,
            "p2": 2 * self.ub2,
        }

    def get_obs(self) -> dict:
        gt_par = self.get_gt_par()
        return {
            "s0": gt_par["p0"],
            "s1": gt_par["p1"],
            "s2": 500,
            "s3": 0.5,
            "s4": 2 * self.ub2,
        }

    def get_ana_args(self) -> Dict[str, Any]:
        return {"population_size": 1000, "max_total_nr_simulations": 50000}

    def get_id(self) -> str:
        return "core"
