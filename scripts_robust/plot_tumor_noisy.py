import os
import matplotlib.pyplot as plt

import slad
import pyabc

pyabc.settings.set_figure_params("pyabc")

distance_names = [
    #"Euclidean",
    #"Manhattan",
    "Calibrated__Euclidean__mad",
    #"Calibrated__Manhattan__mad",
    "Adaptive__Euclidean__mad",
    "Adaptive__Manhattan__mad",
    #"Adaptive__Euclidean__cmad",
    #"Adaptive__Manhattan__cmad",
    #"Adaptive__Euclidean__mad_or_cmad",
    "Adaptive__Manhattan__mad_or_cmad",
    #"Info__Linear__Manhattan__mad_or_cmad",
    #"Info__Linear__Manhattan__mad_or_cmad__Subset",
]

pretty_distances = {
    "Calibrated__Euclidean__mad": "L2 + Calib. + MAD",
    "Adaptive__Euclidean__mad": "L2 + Adap. + MAD",
    "Adaptive__Manhattan__mad": "L1 + Adap. + MAD",
    "Adaptive__Manhattan__mad_or_cmad": "L1 + Adap. + (C)MAD",
    "Info__Linear__Manhattan__mad_or_cmad": "L1 + Adap. + (C)MAD + Info",
    "Info__Linear__Manhattan__mad_or_cmad__Subset": "L1 + Adap. + (C)MAD + Info + Subset",
}

data_dir = "data_robust"
n_rep = 1

for problem_type in [
    "tumor",
]:
    for i_rep in range(n_rep):
        for kwargs in [
            {'noisy': True, 'frac_error': 0},
            {'noisy': True, 'frac_error': 0.1},
        ]:
            if problem_type == "tumor":
                problem = slad.TumorErrorProblem(**kwargs)

            slad.plot_1d_kdes_integrated(problem, distance_names, data_dir=data_dir, problem_suff=f"_{i_rep}", pretty_labels=pretty_distances)
            plt.savefig(f"plot_robust/tumor2d/{problem.get_id()}_{i_rep}_1d_kdes_integrated.png")

            slad.plot_cis(problem, distance_names, data_dir=data_dir, problem_suff=f"_{i_rep}")
            plt.savefig(f"plot_robust/tumor2d/{problem.get_id()}_{i_rep}_cis.png")

            slad.plot_1d_kdes(problem, distance_names, data_dir=data_dir, problem_suff=f"_{i_rep}")
            plt.savefig(f"plot_robust/tumor2d/{problem.get_id()}_{i_rep}_1d_kdes.png")
