import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

import slad
import pyabc

pyabc.settings.set_figure_params("pyabc")

distance_names = [
    #"Euclidean",
    "Calibrated__Euclidean__mad",
    "Adaptive__Euclidean__mad",
    "Adaptive__Euclidean__cmad",
    "Adaptive__Euclidean__mad_or_cmad",
    #"Manhattan",
    "Calibrated__Manhattan__mad",
    "Adaptive__Manhattan__mad",
    "Adaptive__Manhattan__cmad",
    "Adaptive__Manhattan__mad_or_cmad",
    #"Info__Linear__Manhattan__mad_or_cmad",
    #"Info__Linear__Manhattan__mad_or_cmad__All",
    #"Info__Linear__Manhattan__mad_or_cmad__Subset",
    #"Info__Linear__Manhattan__mad_or_cmad__All__Late",
    #"Info__Linear__Manhattan__mad_or_cmad__Subset__Late",
]

distance_labels = {
    "Euclidean": "L2",
    "Calibrated__Euclidean__mad": "L2 + Cal. + MAD",
    "Adaptive__Euclidean__mad": "L2 + Adap. + MAD",
    "Adaptive__Euclidean__cmad": "L2 + Adap. + CMAD",
    "Adaptive__Euclidean__mad_or_cmad": "L2 + Adap. + (C)MAD",
    "Manhattan": "L1",
    "Calibrated__Manhattan__mad": "L1 + Cal. + MAD",
    "Adaptive__Manhattan__mad": "L1 + Adap. + MAD",
    "Adaptive__Manhattan__cmad": "L1 + Adap. + CMAD",
    "Adaptive__Manhattan__mad_or_cmad": "L1 + Adap. + (C)MAD",
    "Info__Linear__Manhattan__mad_or_cmad": "L1 + Adap. + (C)MAD + Info",
    "Info__Linear__Manhattan__mad_or_cmad__All": "L1 + Adap. + (C)MAD + Info + All",
    "Info__Linear__Manhattan__mad_or_cmad__Subset": "L1 + Adap. + (C)AMD + Info + Subset",
    "Info__Linear__Manhattan__mad_or_cmad__All__Late": "L1 + Adap. + (C)MAD + Info + All + Late",
    "Info__Linear__Manhattan__mad_or_cmad__Subset__Late": "L1 + Adap. + (C)MAD + Info + Subset + Late",
}

problem_labels = {
    "uninf": "Uninformative",
    "gaussian": "Replicates",
    "gk": "GK",
    "lv": "Lotka-Volterra",
    "cr-zero": "CR-Zero",
    "cr-swap": "CR-Swap",
}

data_dir = "data_robust"
n_rep = 20


def plot_rmse(n_par, means, stds, problem_type, log: bool):
    n_dist = len(distance_names)
    # plot
    fig, axes = plt.subplots(
        nrows=1, ncols=n_par,
        figsize=(4 + n_par*2, n_dist * 0.5))
    if n_par == 1:
        axes = [axes]
    for i_par, key in enumerate(gt_par.keys()):
        ax = axes[i_par]
        ys = np.arange(n_dist)
        if i_par == 0:
            ax.set_yticks(ys)
            ax.set_yticklabels([distance_labels[name] for name in distance_names])
        else:
            ax.set_yticks([])
        ax.invert_yaxis()
        ax.barh(ys - 0.2, means[:, 0, i_par],
                xerr=stds[:, 0, i_par], color='C0', height=0.4)
        ax.barh(ys + 0.2, means[:, 1, i_par],
                xerr=stds[:, 0, i_par], color='C1', height=0.4)
        # add value
        for i_dist in range(n_dist):
            for i in [0, 1]:
                if log:
                    pos_x = means[i_dist, i, i_par] * 1.4
                else:
                    pos_x = means[i_dist, i, i_par] + 0.2
                ax.text(pos_x, i_dist - (-1)**i * 0.2,
                        f"{means[i_dist, i, i_par]:.3f}",
                        color=f'C{i}', fontsize="xx-small",
                        verticalalignment="center")
        #ax.set_xscale("log")
        ax.set_xlabel("RMSE")
        ax.set_title(key)
        ax.axhline(y=3.5, color="grey", linestyle="--")
        #ax.axhline(y=9.5, color="grey", linestyle="--")

    fig.suptitle(problem_labels[problem_type])
    fig.tight_layout()
    plt.savefig(f"plot_robust/rmse_{problem_type}.png")
    if log:
        for ax in axes:
            ax.set_xscale("log")
    fig.tight_layout()
    if log:
        plt.savefig(f"plot_robust/rmse_{problem_type}_log.png")
    else:
        plt.savefig(f"plot_robust/rmse_{problem_type}.png")


for problem_type in [
    "uninf",
    "gaussian",
    "gk",
    "lv",
    "cr-zero",
    #"cr-swap",
]:
    print(problem_type)

    if problem_type == "uninf":
        problem = slad.UninfErrorProblem()
    elif problem_type == "gaussian":
        problem = slad.GaussianErrorProblem()
    elif problem_type == "gk":
        problem = slad.PrangleGKErrorProblem()
    elif problem_type == "lv":
        problem = slad.PrangleLVErrorProblem()
    elif problem_type == "cr-zero":
        problem = slad.CRErrorZeroProblem()
    elif problem_type == "cr-swap":
        problem = slad.CRErrorSwapProblem()

    gt_par = problem.get_gt_par()

    n_par = len(gt_par)
    n_dist = len(distance_names)

    vals = np.full(shape=(n_dist, 2, n_par, n_rep), fill_value=np.nan)

    for i_mode, kwargs in enumerate([{'n_obs_error': 0}, {}]):

        if problem_type == "uninf":
            problem = slad.UninfErrorProblem(**kwargs)
        elif problem_type == "gaussian":
            problem = slad.GaussianErrorProblem(**kwargs)
        elif problem_type == "gk":
            problem = slad.PrangleGKErrorProblem(**kwargs)
        elif problem_type == "lv":        
            problem = slad.PrangleLVErrorProblem(**kwargs)
        elif problem_type == "cr-zero":
            problem = slad.CRErrorZeroProblem(**kwargs)
        elif problem_type == "cr-swap":
            problem = slad.CRErrorSwapProblem(**kwargs)

        for i_dist, distance_name in enumerate(distance_names):
            print(kwargs, distance_name)
            for i_rep in range(n_rep):
                h = pyabc.History(
                    f"sqlite:///data_robust/{problem.get_id()}_{i_rep}/db_{distance_name}.db",
                    create=False)

                df, w = h.get_distribution(t=h.max_t)
                vals[i_dist, i_mode, :, i_rep] = np.array(
                    [pyabc.weighted_rmse(df[key], w, gt_par[key])
                     for key in gt_par])

                # print(vals[i_dist, i_mode, :, i_rep])

    means = np.mean(vals, axis=3)
    stds = np.std(vals, axis=3)
    #means = np.median(vals, axis=3)
    #stds = stats.median_abs_deviation(vals, axis=3)
    #print(vals[:, 0, :, :])
    #print(vals[:, 1, :, :])
    #print(means)
    #print(stds)

    plot_rmse(n_par=n_par, means=means, stds=stds, problem_type=problem_type, log=True)
    plot_rmse(n_par=n_par, means=means, stds=stds, problem_type=problem_type, log=False)
