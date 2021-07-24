import os
import numpy as np
import matplotlib.pyplot as plt

import slad
import pyabc

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


def plot_sumstats(
    h, data, problem_type, distance_name, arr_ax,
    show_distribution=True, show_mean=True,
):
    color = slad.C.distance_colors[distance_name]
    def f_plot(sum_stat, weight, arr_ax, **kwargs):
        for i, key in enumerate(sum_stat.keys()):
            arr_ax[i].plot(sum_stat[key], '-', color=color, alpha=5*1/5e2)

    def f_plot_mean(sum_stats, weights, arr_ax, **kwargs):
        aggregated = {}
        for key in sum_stats[0].keys():
            aggregated[key] = (np.array([sum_stat[key] for sum_stat in sum_stats]) \
                               * np.array(weights).reshape((-1,1))).sum(axis=0)
        for i, key in enumerate(aggregated.keys()):
            arr_ax[i].plot(aggregated[key], '-', color=color, alpha=1,
                           label=f"{slad.C.distance_labels_short[distance_name]}")

    if not show_distribution:
        f_plot = None
    if not show_mean:
        f_plot_mean = None

    pyabc.visualization.plot_data_callback(h, f_plot, f_plot_mean, ax=arr_ax)

fig = plt.figure(figsize=(14, 3), constrained_layout=True)

arr_ax = fig.subplots(nrows=1, ncols=3)

problem_type = "tumor"
problem = slad.TumorErrorProblem(noisy=True)

dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(dir, "..", "data_robust", f"{problem.get_id()}_0_p200")
data = load_data(problem, data_dir)

for distance_name in distance_names:
    h = pyabc.History(
        "sqlite:///" + os.path.join(data_dir, f"db_{distance_name}.db"),
        create=False)
    plot_sumstats(h, data, problem_type, distance_name, arr_ax, show_mean=False)

for distance_name in distance_names:
    h = pyabc.History(
        "sqlite:///" + os.path.join(data_dir, f"db_{distance_name}.db"),
        create=False)
    plot_sumstats(h, data, problem_type, distance_name, arr_ax, show_distribution=False)

# data and labels
for i, key in enumerate(data.keys()):
    arr_ax[i].plot(data[key], 'x', color="black", label='Observed data')
    arr_ax[i].set_ylabel(slad.C.data_ylabels[problem_type][key])
    arr_ax[i].set_xlabel(slad.C.data_xlabels[problem_type][key])

arr_ax[-1].legend()

for fmt in ["svg", "png"]:
    plt.savefig(f"figures_robust/figure_tumor_error_fits_base.{fmt}", format=fmt)

