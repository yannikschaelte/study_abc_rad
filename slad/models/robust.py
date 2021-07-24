"""Introduction of outliers to problems."""

from typing import Callable
import numpy as np

import pyabc

from .base import Problem
from .prangle2015 import PrangleGKProblem, PrangleLVProblem
from .cr import CRProblem
from .tumor import TumorProblem


class GaussianErrorProblem(Problem):
    """Informative replicates, some of which are outliers."""

    def __init__(self, n_obs: int = 10, n_obs_error: int = 2):
        self.n_obs: int = n_obs
        self.n_obs_error: int = n_obs_error

    def get_prior(self) -> pyabc.Distribution:
        return pyabc.Distribution(p0=pyabc.RV("uniform", 0, 10))

    def get_prior_bounds(self) -> dict:
        return {"p0": (0, 10)}

    def get_obs(self) -> dict:
        model = self.get_model()
        obs = model(self.get_gt_par())
        obs = self.errorfy(obs)
        return obs

    def errorfy(self, obs: dict) -> dict:
        obs["y"][: self.n_obs_error] = 0
        return obs

    def get_gt_par(self) -> dict:
        return {"p0": 6.0}

    def get_id(self) -> str:
        return f"gaussian_error_{self.n_obs}_{self.n_obs_error}"

    def get_model(self) -> Callable:
        def model(p: dict):
            return {"y": p["p0"] + 0.2 * np.random.normal(size=self.n_obs)}

        return model


class UninfErrorProblem(Problem):
    """Single uninformative outliers."""

    def __init__(self, n_obs_error: int = 1):
        self.n_obs_error: int = n_obs_error

    def get_prior(self) -> pyabc.Distribution:
        return pyabc.Distribution(p0=pyabc.RV("uniform", 0, 10))

    def get_prior_bounds(self) -> dict:
        return {"p0": (0, 10)}

    def get_obs(self) -> dict:
        model = self.get_model()
        obs = model(self.get_gt_par())
        obs = self.errorfy(obs)
        return obs

    def errorfy(self, obs: dict) -> dict:
        if self.n_obs_error > 0:
            obs["y"][-1] = 7
        return obs

    def get_gt_par(self) -> dict:
        return {"p0": 5.0}

    def get_id(self) -> str:
        return f"uninf_{self.n_obs_error}"

    def get_model(self) -> Callable:
        def model(p) -> dict:
            sim = p["p0"] + 1 * np.random.normal(size=11)
            # last one is different
            sim[-1] = 5 + 0.1 * np.random.normal()
            return {"y": sim}

        return model


class PrangleGKErrorProblem(PrangleGKProblem):
    """Introduction of outliers to the GK problem by Prangle et al.."""

    def __init__(self, n_obs_error: int = 1):
        self.n_obs_error: int = n_obs_error

    def get_obs(self) -> dict:
        obs = super().get_obs()
        obs = self.errorfy(obs)
        return obs

    def errorfy(self, obs: dict) -> dict:
        if self.n_obs_error > 0:
            err_ixs = np.random.permutation(len(obs["y"]))[: self.n_obs_error]
            obs["y"][err_ixs] = 0
        return obs

    def get_id(self) -> str:
        return f"prangle_gk_{self.n_obs_error}"


class PrangleLVErrorProblem(PrangleLVProblem):
    """Introduction of outliers to the Lotka-Volterra problem by
    Prangle et al.
    """

    def __init__(self, n_obs_error: int = 6):
        self.n_obs_error: int = n_obs_error

    def get_obs(self) -> dict:
        obs = super().get_obs()
        obs = self.errorfy(obs)
        return obs

    def errorfy(self, obs: dict) -> dict:
        if self.n_obs_error > 0:
            if self.n_obs_error == 3:
                err_ixs = np.random.permutation(len(obs["y"][:, 0]))[: self.n_obs_error]
            # obs["y"][err_ixs, :] = 0  # - obs["y"][err_ixs, : ]
                obs["y"][err_ixs, :] = -obs["y"][err_ixs, :]
            elif self.n_obs_error == 4:
                err_ixs = np.random.permutation(len(obs["y"][:, 0]))[: 3]
                obs["y"][err_ixs, :] = 0
            elif self.n_obs_error == 5:
                err_ixs = np.random.permutation(len(obs["y"][:, 0]))[: 3]
                obs["y"][err_ixs, :] = np.flip(obs["y"][err_ixs, :])
            elif self.n_obs_error == 6:
                err_ixs = np.random.permutation(len(obs["y"][:, 0]))[: 3]
                obs["y"][err_ixs, :] *= 10
            # for err_ix in err_ixs:
            #    obs["y"][err_ix, 0], obs["y"][err_ix, 1] = obs["y"][err_ix, 1], obs["y"][err_ix, 0]
        return obs

    def get_id(self) -> str:
        return f"{super().get_id()}_{self.n_obs_error}"


class CRErrorZeroProblem(CRProblem):
    """Introduction of zero-value outliers to the conversion reaction problem."""

    def __init__(self, noise_std: float = 0.02, n_obs_error: int = 1):
        super().__init__(noise_std=noise_std)
        self.n_obs_error: int = n_obs_error

    def get_obs(self) -> dict:
        obs = super().get_obs()
        obs = self.errorfy(obs)
        return obs

    def errorfy(self, obs: dict) -> dict:
        if self.n_obs_error == 0:
            return obs

        obs["y"][np.random.permutation(len(obs["y"]))[: self.n_obs_error]] = 0
        return obs

    def get_id(self) -> str:
        return f"CRZero_{self.n_obs_error}"


class CRErrorSwapProblem(CRProblem):
    """Alternative outliers in the conversion reaction problem by observable
    swaps.
    """

    def __init__(self, noise_std: float = 0.02, n_obs_error: int = 1):
        super().__init__(noise_std=noise_std)
        self.n_obs_error: int = n_obs_error

    def get_obs(self) -> dict:
        obs = super().get_obs()
        obs = self.errorfy(obs)
        return obs

    def errorfy(self, obs: dict) -> dict:
        if self.n_obs_error == 0:
            return obs

        ix0, ix1 = np.random.permutation(len(obs["y"]))[:2]
        obs["y"][ix0], obs["y"][ix1] = obs["y"][ix1], obs["y"][ix0]
        # obs["y"][1], obs["y"][-2] = obs["y"][-2], obs["y"][1]
        return obs

    def get_id(self) -> str:
        return f"CRSwap_{self.n_obs_error}"


class TumorErrorProblem(TumorProblem):
    """Introduction of outliers to the tumor problem."""

    def __init__(self, obs_rep: int = 1, noisy: bool = False, frac_error: float = 0.1):
        super().__init__(obs_rep=obs_rep, noisy=noisy)
        self.frac_error: float = frac_error

    def get_obs(self) -> dict:
        obs = super().get_obs()
        obs = self.errorfy(obs)
        return obs

    def errorfy(self, obs: dict) -> dict:
        if self.frac_error > 0:
            for key in obs.keys():
                if key == "proliferation_profile":
                    n_obs = 22
                elif key == "extra_cellular_matrix_profile":
                    n_obs = 65
                elif key == "growth_curve":
                    n_obs = 20
                else:
                    raise ValueError(f"key {key} does not exist")

                n_err = int(self.frac_error * n_obs)
                for i_err in range(n_err):
                    obs[key][i_err * 3 + 2], obs[key][n_obs - (i_err * 3 + 2) - 1] = (
                        obs[key][n_obs - (i_err * 3 + 2) - 1],
                        obs[key][i_err * 3 + 2],
                    )
                # err_ixs = np.random.permutation(n_obs)[:n_err]
                # obs[key][err_ixs] = np.random.permutation(obs[key][err_ixs])
        return obs

    def get_id(self) -> str:
        return f"{super().get_id()}_{self.frac_error}"
