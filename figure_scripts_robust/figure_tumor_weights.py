import os
import numpy as np
import matplotlib.pyplot as plt

import slad
import pyabc
from pyabc.storage import load_dict_from_json

pyabc.settings.set_figure_params("pyabc")

distance_names = [
    "Calibrated__Euclidean__mad",
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
arr_ax = fig.subplots(nrows=1, ncols=3)

colors = [slad.C.distance_colors[dname] for dname in distance_names]

frac_error = 0.1
frac_error = 0
problem = slad.TumorErrorProblem(noisy=True, frac_error=frac_error)
problem_type = "tumor"

n_data = len(slad.C.data_ylabels[problem_type])

dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(dir, "..", "data_robust", f"{problem.get_id()}_0_p200")
data = load_data(problem, data_dir)

scale_log_files = []
for i_dist, distance_name in enumerate(distance_names):
    scale_log_file = os.path.join(data_dir, f"log_scale_{distance_name}.json")
    scale_log_files.append(scale_log_file)

sumstat_keys_arr = [
    [f"growth_curve:{i}" for i in range(20)],
    [f"extra_cellular_matrix_profile:{i}" for i in range(65)],
    [f"proliferation_profile:{i}" for i in range(65)],
]

for i_sumstat, sumstat_keys in enumerate(sumstat_keys_arr):
    pyabc.visualization.plot_distance_weights(
        scale_log_files,
        labels=[slad.C.distance_labels_short[dname] for dname in distance_names],
        ax=arr_ax[i_sumstat], colors=colors,
        keys_as_labels=False,
        keys=sumstat_keys,
    )

for i, key in enumerate(data.keys()):
    arr_ax[i].set_ylabel(slad.C.data_ylabels[problem_type][key])
    arr_ax[i].set_xlabel(slad.C.data_xlabels[problem_type][key])

for ax in arr_ax:
    ax.get_legend().remove()
arr_ax[-1].legend()

for fmt in ["svg", "png"]:
    plt.savefig(f"figures_robust/figure_tumor_weights_{frac_error}.{fmt}", format=fmt)
