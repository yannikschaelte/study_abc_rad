import os
import numpy as np
import matplotlib.pyplot as plt

import slad
import pyabc
from pyabc.storage import load_dict_from_json

pyabc.settings.set_figure_params("pyabc")

distance_names = [
    #"Euclidean",
    #"Manhattan",
    "Calibrated__Euclidean__mad",
    #"Calibrated__Manhattan__mad",
    "Adaptive__Euclidean__mad",
    "Adaptive__Manhattan__mad",
    #"Adaptive__Euclidean__cmad",
    "Adaptive__Manhattan__cmad",
    #"Adaptive__Euclidean__mad_or_cmad",
    "Adaptive__Manhattan__mad_or_cmad",
    #"Info__Linear__Manhattan__mad_or_cmad",
    #"Info__Linear__Manhattan__mad_or_cmad__Subset",
]

data_dir = "data_robust"
n_rep = 20

for problem_type in [
    "uninf",
    "gaussian",
    "gk",
    "lv",
    "CRZero",
    #"CRSwap",
]:
    for i_rep in range(n_rep):
        for kwargs in [{'n_obs_error': 0}, {}]:
            print(problem_type, i_rep, kwargs)
            if problem_type == "uninf":
                problem = slad.UninfErrorProblem(**kwargs)
            elif problem_type == "gaussian":
                problem = slad.GaussianErrorProblem(**kwargs)
            elif problem_type == "gk":
                problem = slad.PrangleGKErrorProblem(**kwargs)
            elif problem_type == "lv":
                problem = slad.PrangleLVErrorProblem(**kwargs)
            elif problem_type == "CRZero":
                problem = slad.CRErrorZeroProblem(**kwargs)
            elif problem_type == "CRSwap":
                problem = slad.CRErrorSwapProblem(**kwargs)

            dir = os.path.dirname(os.path.realpath(__file__))
            data_dir = os.path.join(dir, "..", "data_robust", f"{problem.get_id()}_{i_rep}")

            fig, ax = plt.subplots()

            scale_log_files = []

            for i_dist, distance_name in enumerate(distance_names):
                scale_log_file = os.path.join(data_dir, f"log_scale_{distance_name}.json")
                scale_log_files.append(scale_log_file)

            pyabc.visualization.plot_distance_weights(scale_log_files, labels=distance_names, ax=ax)
            plt.savefig(f"plot_robust/weight/{problem.get_id()}_{i_rep}.png")
            plt.close()
