import os
import numpy as np
import matplotlib.pyplot as plt

import slad
import pyabc

pyabc.settings.set_figure_params("pyabc")

distance_names = [
    #"Euclidean",
    #"Manhattan",
    "Calibrated__Euclidean__mad",
    "Calibrated__Manhattan__mad",
    "Adaptive__Euclidean__mad",
    "Adaptive__Manhattan__mad",
    "Adaptive__Euclidean__cmad",
    "Adaptive__Manhattan__cmad",
    "Adaptive__Euclidean__mad_or_cmad",
    "Adaptive__Manhattan__mad_or_cmad",
    #"Info__Linear__Manhattan__mad_or_cmad",
    #"Info__Linear__Manhattan__mad_or_cmad__Subset",
]

def load_data(problem, data_dir):
    data = {}
    for key in problem.get_y_keys():
        data[key] = np.loadtxt(os.path.join(data_dir, f"data_{key}.csv"), delimiter=",")
        if data[key].size == 1:
            data[key] = float(data[key])
    return data


data_ylabels = {
    "tumor": {
        "growth_curve": "Spheroid radius [$\\mu m$]",
        "proliferation_profile": "Frac. proliferating cells",
        "extra_cellular_matrix_profile": "ECM intensity"},
    "uninf": {
        "y": "Value"},
    "gaussian": {
        "y": "Value"},
    "gk": {
        "y": "Value"},
    "lv": {
        "Prey": "Number",
        "Predator": "Number"},
    "CRZero": {
        "y": "Species B"},
    "CRSwap": {
        "y": "Species B"},
}
data_xlabels = {
    "tumor": {
        "growth_curve": "Time [$d$]",
        "proliferation_profile": "Distance to rim [$\\mu m$]",
        "extra_cellular_matrix_profile": "Distance to rim [$\\mu m$]"},
    "uninf": {
        "y": "Coordinate"},
    "gaussian": {
        "y": "Replicate"},
    "gk": {
        "y": "Order statistic"},
    "lv": {
        "Prey": "Time [au]",
        "Predator": "Time [au]"},
    "CRZero": {
        "y": "Time [au]"},
    "CRSwap": {
        "y": "Time [au]"},
}


def plot_sumstats(h, title, data, problem_type, arr_ax):
    def f_plot(sum_stat, weight, arr_ax, **kwargs):
        for i, key in enumerate(sum_stat.keys()):
            arr_ax[i].plot(sum_stat[key], '-', color='grey', alpha=5*1/5e2)#min(20*weight, 1))
    def f_plot_lv(sum_stat, weight, arr_ax, **kwargs):
        arr_ax[0].plot(sum_stat["y"][:, 0], '-', color='grey', alpha=5*1/5e2)
        arr_ax[1].plot(sum_stat["y"][:, 1], '-', color='grey', alpha=5*1/5e2)
        
    def f_plot_mean(sum_stats, weights, arr_ax, **kwargs):
        aggregated = {}
        for key in sum_stats[0].keys():
            aggregated[key] = (np.array([sum_stat[key] for sum_stat in sum_stats]) \
                               * np.array(weights).reshape((-1,1))).sum(axis=0)
        for i, key in enumerate(aggregated.keys()):
            arr_ax[i].plot(aggregated[key], '-', color='C3', alpha=1, label='ABC posterior mean')
    def f_plot_mean_lv(sum_stats, weights, arr_ax, **kwargs):
        aggregated = {}
        aggregated["Prey"] = (np.array([sum_stat["y"][:, 0] for sum_stat in sum_stats]) \
                              * np.array(weights).reshape((-1,1))).sum(axis=0)
        aggregated["Predator"] = (np.array([sum_stat["y"][:, 1] for sum_stat in sum_stats]) \
                                  * np.array(weights).reshape((-1,1))).sum(axis=0)
        arr_ax[0].plot(aggregated["Prey"], '-', color='C3', alpha=1, label='ABC posterior mean')
        arr_ax[1].plot(aggregated["Predator"], '-', color='C3', alpha=1, label='ABC posterior mean')

    if problem_type != "lv":
        pyabc.visualization.plot_data_callback(h, f_plot, f_plot_mean, ax=arr_ax)
    else:
        pyabc.visualization.plot_data_callback(h, f_plot_lv, f_plot_mean_lv, ax=arr_ax)

    if problem_type != "lv":
        for i, key in enumerate(data.keys()):
            arr_ax[i].plot(data[key], 'x', color='C9', label='data')
            arr_ax[i].set_ylabel(data_ylabels[problem_type][key])
            arr_ax[i].set_xlabel(data_xlabels[problem_type][key])
    else:
        arr_ax[0].plot(data["y"][:, 0], 'x', color='C9', label='data')
        arr_ax[0].set_ylabel("# Prey")
        arr_ax[0].set_xlabel("Time [au]")
        arr_ax[1].plot(data["y"][:, 1], 'x', color='C9', label='data')
        arr_ax[1].set_ylabel("# Predator")
        arr_ax[1].set_xlabel("Time [au]")

    if len(data) == 1:
        arr_ax[0].set_title(title)
    else:
        arr_ax[0].set_title(title)
        
    arr_ax[-1].legend()
    for ax in arr_ax:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

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
            data = load_data(problem, data_dir)
            
            n_data = len(data)
            if problem_type == "lv":
                n_data = 2
            fig, arr_ax = plt.subplots(len(distance_names), n_data)

            for i_dist, distance_name in enumerate(distance_names):
                print(problem_type, i_rep, kwargs, distance_name)
                h = pyabc.History(
                    "sqlite:///" + os.path.join(data_dir, f"db_{distance_name}.db"), create=False)
                axes = arr_ax[i_dist]
                if n_data == 1:
                    axes = [axes]
                plot_sumstats(h, distance_name, data, problem_type, axes)
            
            #fig.suptitle(title)
            fig.set_size_inches((5 * n_data, 4 * len(distance_names)))
            fig.tight_layout()
            #fig.tight_layout(rect=[0, 0, 1, 0.95])

            plt.savefig(f"plot_robust/fit/fit_{problem.get_id()}_{i_rep}.png")
            plt.close()

