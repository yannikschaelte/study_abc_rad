import os
import numpy as np
import matplotlib.pyplot as plt

import slad
import pyabc
from pyabc.storage import load_dict_from_json

pyabc.settings.set_figure_params("pyabc")

distance_names = [
    #"Calibrated__Euclidean__mad",
    "Adaptive__Euclidean__mad",
    "Adaptive__Manhattan__mad",
    "Adaptive__Manhattan__mad_or_cmad",
]

def load_data(problem, data_dir):
    data = {}
    for key in problem.get_y_keys():
        data[key] = np.loadtxt(os.path.join(data_dir, f"data_{key}.csv"), delimiter=",")
        if data[key].size == 1:
            data[key] = float(data[key])
    return data

fig = plt.figure(figsize=(14, 2.5), constrained_layout=True)
width_ratios = [3, 3, 3, 4.5]
subfigs = fig.subfigures(nrows=1, ncols=4, wspace=0.01, width_ratios=width_ratios)

configs = [
    ("gaussian", 19),
    ("cr-zero", 2),
    ("gk", 16),
    ("lv", 1),
]

for i_subfig, (subfig, (problem_type, i_rep)) in enumerate(zip(subfigs, configs)):
    colors = [slad.C.distance_colors[dname] for dname in distance_names]

    kwargs = {}
    if problem_type == "gaussian":
        problem = slad.GaussianErrorProblem(**kwargs)
    elif problem_type == "gk":
        problem = slad.PrangleGKErrorProblem(**kwargs)
    elif problem_type == "lv":
        problem = slad.PrangleLVErrorProblem(**kwargs)
    elif problem_type == "cr-zero":
        problem = slad.CRErrorZeroProblem(**kwargs)
    elif problem_type == "cr-swap":
        problem = slad.CRErrorSwapProblem(**kwargs)

    n_data = len(slad.C.data_ylabels[problem_type])
    if problem_type == "lv":
        n_data = 2
    arr_ax = subfig.subplots(nrows=1, ncols=n_data)

    if n_data == 1:
        arr_ax = [arr_ax]

    dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(dir, "..", "data_robust", f"{problem.get_id()}_{i_rep}")
    data = load_data(problem, data_dir)

    scale_log_files = []
    for i_dist, distance_name in enumerate(distance_names):
        scale_log_file = os.path.join(data_dir, f"log_scale_{distance_name}.json")
        scale_log_files.append(scale_log_file)

    if problem_type != "lv":
        pyabc.visualization.plot_distance_weights(
            scale_log_files,
            labels=[slad.C.distance_labels_short[dname] for dname in distance_names],
            ax=arr_ax[0], colors=colors,
            keys_as_labels=False)
    else:
        keys = [f"y:{i}" for i in range(0, 16)]
        pyabc.visualization.plot_distance_weights(
            scale_log_files,
            labels=[slad.C.distance_labels_short[dname] for dname in distance_names],
            ax=arr_ax[0], colors=colors, keys=keys, keys_as_labels=False)
        keys = [f"y:{i}" for i in range(16, 32)]
        pyabc.visualization.plot_distance_weights(
            scale_log_files,
            labels=[slad.C.distance_labels_short[dname] for dname in distance_names],
            ax=arr_ax[1], colors=colors, keys=keys, keys_as_labels=False)

    # labels
    if problem_type != "lv":
        for i, key in enumerate(data.keys()):
            arr_ax[i].set_title(slad.C.data_ylabels[problem_type][key])
            arr_ax[i].set_xlabel(slad.C.data_xlabels[problem_type][key])
    else:
        arr_ax[0].set_title("# Prey")
        arr_ax[0].set_xlabel("Time [au]")
        arr_ax[1].set_title("# Predator")
        arr_ax[1].set_xlabel("Time [au]")

    for ax in arr_ax:
        ax.get_legend().remove()
    if i_subfig == 0:
        arr_ax[0].legend()


    subfig.suptitle(slad.C.problem_labels[problem_type])

plt.savefig(f"figures_robust/figure_weights.svg", format="svg")
