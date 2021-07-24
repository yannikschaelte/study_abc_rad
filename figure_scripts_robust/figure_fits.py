import os
import numpy as np
import matplotlib.pyplot as plt

import slad
import pyabc

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


def plot_sumstats(
    h, data, problem_type, distance_name, arr_ax,
    show_distribution=True, show_mean=True,
):
    color = slad.C.distance_colors[distance_name]
    def f_plot(sum_stat, weight, arr_ax, **kwargs):
        for i, key in enumerate(sum_stat.keys()):
            arr_ax[i].plot(sum_stat[key], '-', color=color, alpha=5*1/5e2)
    def f_plot_lv(sum_stat, weight, arr_ax, **kwargs):
        arr_ax[0].plot(sum_stat["y"][:, 0], '-', color=color, alpha=5*1/5e2)
        arr_ax[1].plot(sum_stat["y"][:, 1], '-', color=color, alpha=5*1/5e2)

    def f_plot_mean(sum_stats, weights, arr_ax, **kwargs):
        aggregated = {}
        for key in sum_stats[0].keys():
            aggregated[key] = (np.array([sum_stat[key] for sum_stat in sum_stats]) \
                               * np.array(weights).reshape((-1,1))).sum(axis=0)
        for i, key in enumerate(aggregated.keys()):
            arr_ax[i].plot(aggregated[key], '-', color=color, alpha=1,
                           label=f"{slad.C.distance_labels_short[distance_name]}")
    def f_plot_mean_lv(sum_stats, weights, arr_ax, **kwargs):
        aggregated = {}
        aggregated["Prey"] = (np.array([sum_stat["y"][:, 0] for sum_stat in sum_stats]) \
                              * np.array(weights).reshape((-1,1))).sum(axis=0)
        aggregated["Predator"] = (np.array([sum_stat["y"][:, 1] for sum_stat in sum_stats]) \
                                  * np.array(weights).reshape((-1,1))).sum(axis=0)
        arr_ax[0].plot(
            aggregated["Prey"], '-', color=color, alpha=1,
            label=f"{slad.C.distance_labels_short[distance_name]}")
        arr_ax[1].plot(
            aggregated["Predator"], '-', color=color, alpha=1,
            label=f"{slad.C.distance_labels_short[distance_name]}")

    if problem_type == "lv":
        f_plot = f_plot_lv
        f_plot_mean = f_plot_mean_lv

    if not show_distribution:
        f_plot = None
    if not show_mean:
        f_plot_mean = None

    pyabc.visualization.plot_data_callback(h, f_plot, f_plot_mean, ax=arr_ax)

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
    if problem_type != "lv":
        for i, key in enumerate(data.keys()):
            arr_ax[i].plot(data[key], 'x', color="black", label='Observed data')
            arr_ax[i].set_ylabel(slad.C.data_ylabels[problem_type][key])
            arr_ax[i].set_xlabel(slad.C.data_xlabels[problem_type][key])
    else:
        arr_ax[0].plot(data["y"][:, 0], 'x', color="black", label='Observed data')
        arr_ax[0].set_ylabel("# Prey")
        arr_ax[0].set_xlabel("Time [au]")
        arr_ax[1].plot(data["y"][:, 1], 'x', color="black", label='Observed data')
        arr_ax[1].set_ylabel("# Predator")
        arr_ax[1].set_xlabel("Time [au]")

    if i_subfig == 0:
        arr_ax[0].legend()

    subfig.suptitle(slad.C.problem_labels[problem_type])

plt.savefig(f"figures_robust/figure_fits_base.svg", format="svg")

