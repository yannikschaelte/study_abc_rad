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

def load_data(problem, data_dir):
    data = {}
    for key in problem.get_y_keys():
        data[key] = np.loadtxt(os.path.join(data_dir, f"data_{key}.csv"), delimiter=",")
        if data[key].size == 1:
            data[key] = float(data[key])
    return data


data_ylabels = ["Spheroid radius [$\\mu m$]", "Frac. proliferating cells", "ECM intensity"]
data_xlabels = ["Time [$d$]", "Distance to rim [$\\mu m$]", "Distance to rim [$\\mu m$]"]


def plot_sumstats(h, title, data, id_):
    fig, arr_ax = plt.subplots(1, 3)
    
    def f_plot(sum_stat, weight, arr_ax, **kwargs):
        for i, key in enumerate(sum_stat.keys()):
            arr_ax[i].plot(sum_stat[key], '-', color='grey', alpha=5*1/5e2)#min(20*weight, 1))
        
    def f_plot_mean(sum_stats, weights, arr_ax, **kwargs):
        aggregated = {}
        for key in sum_stats[0].keys():
            aggregated[key] = (np.array([sum_stat[key] for sum_stat in sum_stats]) \
                               * np.array(weights).reshape((-1,1))).sum(axis=0)
        for i, key in enumerate(aggregated.keys()):
            arr_ax[i].plot(aggregated[key], '-', color='C3', alpha=1, label='ABC posterior mean')

    pyabc.visualization.plot_data_callback(h, f_plot, f_plot_mean, ax=arr_ax)

    for i, key in enumerate(data.keys()):
        arr_ax[i].plot(data[key], 'x', color='C9', label='data')
        arr_ax[i].set_ylabel(data_ylabels[i])
        arr_ax[i].set_xlabel(data_xlabels[i])
    arr_ax[0].legend()
    for ax in arr_ax:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    fig.suptitle(title)
    fig.set_size_inches((10, 4))
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig(f"plot_robust/tumor2d/fit_{id_}.png")


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

            dir = os.path.dirname(os.path.realpath(__file__))
            data_dir = os.path.join(dir, "..", "data_robust", f"{problem.get_id()}_0_p200")
            data = load_data(problem, data_dir)

            for distance_name in distance_names:
                h = pyabc.History(
                    "sqlite:///" + os.path.join(data_dir, f"db_{distance_name}.db"), create=False)
                plot_sumstats(h, distance_name, data, f"{problem.get_id()}_{distance_name}")
