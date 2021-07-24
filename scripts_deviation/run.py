import os
import numpy as np
import logging

import pyabc
from pyabc import ABCSMC, RedisEvalParallelSampler
from pyabc.distance import *
from pyabc.sumstat import *
from pyabc.predictor import *

import slad


# for debugging
for logger in ["ABC.Distance", "ABC.Predictor", "ABC.Sumstat"]:
    logging.getLogger(logger).setLevel(logging.DEBUG)

# read cmd line arguments
host, port = slad.read_args()


def get_distance(name: str) -> pyabc.Distance:
    if name == "Euclidean":
        return PNormDistance(p=2)
    if name == "Manhattan":
        return PNormDistance(p=1)

    if name == "Calibrated__Euclidean__mad":
        return AdaptivePNormDistance(p=2, scale_function=mad, fit_scale_ixs={0})
    if name == "Calibrated__Manhattan__mad":
        return AdaptivePNormDistance(p=1, scale_function=mad, fit_scale_ixs={0})

    if name == "Adaptive__Euclidean__mad":
        return AdaptivePNormDistance(p=2, scale_function=mad)
    if name == "Adaptive__Manhattan__mad":
        return AdaptivePNormDistance(p=1, scale_function=mad)

    if name == "Adaptive__Euclidean__cmad":
        return AdaptivePNormDistance(p=2, scale_function=cmad)
    if name == "Adaptive__Manhattan__cmad":
        return AdaptivePNormDistance(p=1, scale_function=cmad)

    if name == "Adaptive__Euclidean__mad_or_cmad":
        return AdaptivePNormDistance(p=2, scale_function=mad_or_cmad)
    if name == "Adaptive__Manhattan__mad_or_cmad":
        return AdaptivePNormDistance(p=1, scale_function=mad_or_cmad)

    raise ValueError()

distance_names = [
    "Adaptive__Euclidean__mad",
    "Adaptive__Manhattan__mad",
    "Adaptive__Manhattan__mad_or_cmad",
]

pop_size = 1000
max_total_sim = 100000

for problem_type in ["uninf", "gaussian"]:
    for deviation in [0, 1, 5, 25, 125]:
        if problem_type == "uninf":
            problem = slad.UninfErrorProblem()
            std = 0.1
            data = 5 * np.ones(11)
            data[-1] = 5 + deviation * std
        elif problem_type == "gaussian":
            problem = slad.GaussianErrorProblem()
            std = 0.2
            data = 6 * np.ones(10)
            data[[0, 1]] = 6 + deviation * std
        else:
            raise ValueError()

        problem_dir = os.path.join("data_deviation", f"{problem.get_id()}_{deviation}")
        os.makedirs(problem_dir, exist_ok=True)

        for distance_name in distance_names:
            db_file = os.path.join(problem_dir, f"db_{distance_name}.db")
            if os.path.exists(db_file):
                continue
            
            acceptor = pyabc.UniformAcceptor(use_complete_history=True)

            abc = ABCSMC(
                problem.get_model(),
                problem.get_prior(),
                get_distance(distance_name),
                population_size=pop_size,
                acceptor=acceptor,
            )
            abc.new("sqlite:///" + db_file, {"y": data})
            abc.run(max_total_nr_simulations=max_total_sim)

print("ABC out")

