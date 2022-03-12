import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import scipy.stats as stats
import pickle
import argparse

import slad
import pyabc

pyabc.settings.set_figure_params("pyabc")

parser = argparse.ArgumentParser()
parser.add_argument("--hist", type=int)
args = parser.parse_args()
use_hist = args.hist
assert use_hist in [0, 1]
print("hist", use_hist)
if use_hist == 1:
    data_dir = "data_hist"
    hist_suf = "_hist"
else:
    data_dir = "data_robust"
    hist_suf = ""

n_rep = 20

fontsize_big = 12
fontsize_medium = 10
fontsize_small = 8
padding = 0.3

tumor_distance_names = [
    "Calibrated__Euclidean__mad",
    "Adaptive__Euclidean__mad",
    "Adaptive__Manhattan__mad",
    "Adaptive__Manhattan__mad_or_cmad",
]

def create_vals(problem_type):
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
    elif problem_type == "tumor":
        problem = slad.TumorErrorProblem()
    else:
        raise ValueError()

    gt_par = problem.get_gt_par()

    n_par = len(gt_par)
    n_dist = len(slad.C.distance_names)

    vals = np.full(shape=(n_dist, 2, n_rep), fill_value=np.nan)

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
        elif problem_type == "tumor":
            if kwargs:
                problem = slad.TumorErrorProblem(noisy=True, frac_error=0)
            else:
                problem = slad.TumorErrorProblem(noisy=True, frac_error=0.1)
        else:
            raise ValueError()

        for i_dist, distance_name in enumerate(slad.C.distance_names):
            if problem_type != "tumor":
                for i_rep in range(n_rep):
                    h = pyabc.History(
                        f"sqlite:///{data_dir}/{problem.get_id()}_{i_rep}/db_{distance_name}.db",
                        create=False)
                    _, w = h.get_distribution(t=h.max_t)
                    vals[i_dist, i_mode, i_rep] = pyabc.effective_sample_size(w)
            else:
                if distance_name in tumor_distance_names:
                    h = pyabc.History(
                        f"sqlite:///{data_dir}/{problem.get_id()}_0/db_{distance_name}.db",
                        create=False)
                    _, w = h.get_distribution(t=h.max_t)
                    vals[i_dist, i_mode, :] = pyabc.effective_sample_size(w)
                else:
                    vals[i_dist, i_mode, :] = 0


    means = np.mean(vals, axis=2)
    stds = np.std(vals, axis=2)

    return means, stds, gt_par


def plot_ess(
    problem_type,
    log: bool,
    axes,
    ylabels: bool,
):
    print(problem_type)
    means, stds, gt_par = create_vals(problem_type)
    n_par = len(gt_par)

    n_dist = len(slad.C.distance_names)
    colors = list(slad.C.distance_colors.values())
    for i_par in [0]:
        ax = axes[i_par]
        ys = np.arange(n_dist)
        if ylabels and i_par == 0:
            ax.set_yticks(np.arange(n_dist))
            ax.set_yticklabels([
                slad.C.distance_labels_short[dname] for dname in slad.C.distance_names],
                fontdict={"fontsize": fontsize_medium},
            )
            ax.yaxis.set_ticks_position("none")
        else:
            ax.set_yticks([])

        ax.invert_yaxis()
        ax.barh(
            ys - 0.2, means[:, 0],
            xerr=stds[:, 0],
            color=colors, alpha=0.3, height=0.4,
            error_kw={"ecolor": "grey", "alpha": 0.5},
        )
        ax.barh(
            ys + 0.2, means[:, 1],
            xerr=stds[:, 1],
            color=colors, alpha=0.8, height=0.4,
            error_kw={"ecolor": "grey", "alpha": 0.5},
        )
        if log:
            ax.set_xscale("log")

        # add value
        for i_dist in range(n_dist):
            for i in [0, 1]:
                max_val = means[:, :].max()
                if log:
                    pos_x = means[i_dist, i] * (1 + 1 / max_val)
                else:
                    pos_x = means[i_dist, i] + (1  + 1 / max_val)
                mean = means[i_dist, i]
                if mean == 0:
                    continue
                ax.text(max_val * 0.9,
                        i_dist - (-1)**i * 0.2,
                        f"{means[i_dist, i]:.1f}",
                        fontdict={"fontsize": fontsize_medium},
                        verticalalignment="center",
                        horizontalalignment="right")

        #ax.set_xlabel("RMSE")
        #ax.set_title(slad.C.parameter_labels[problem_type][key], fontsize=fontsize_medium)
        ax.axhline(y=3.5, color="grey", linestyle="dotted")

        plt.setp(ax.get_xticklabels(), fontsize=fontsize_small)
        plt.setp(ax.get_xminorticklabels(), visible=False)

    axes[0].text(
        0, 1.08, slad.C.problem_labels[problem_type],
        horizontalalignment="left", verticalalignment="bottom",
        transform=axes[0].transAxes, fontsize=fontsize_big,
    )

problem_types = ["uninf", "gaussian", "cr-zero", "gk", "lv", "tumor"]
arr_cols = [1, 1, 1, 1, 1, 1]
fig, axes = plt.subplots(
    nrows=1, ncols=sum(arr_cols), figsize=(12, 5),
    # constrained_layout=True,
)

for i, (problem_type, cols) in enumerate(zip(problem_types, arr_cols)):
    axes_for_problem = axes[sum(arr_cols[:i]):sum(arr_cols[:i+1])]
    plot_ess(
        problem_type=problem_type,
        log=False,
        axes=axes_for_problem,
        ylabels=i==0,
    )

# fig.tight_layout()
plt.subplots_adjust(left=0.12, right=0.99, top=0.89, bottom=0.13)

# x axis label
fig.text(
    0.5, 0.05, "Effective sample size",
    horizontalalignment="center", verticalalignment="center",
    fontsize=fontsize_medium,
)

# legend
legend_elements = [
    Patch(facecolor="grey", alpha=0.3, label="Outlier-free"),
    Patch(facecolor="grey", alpha=0.8, label="Outlier-corrupted"),
]
axes[-1].legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1, -0.07), ncol=2)
#plt.subplots_adjust(bottom=0.1)

for fmt in ["pdf", "png"]:
    plt.savefig(f"figures_robust/figure_ess{hist_suf}.{fmt}", format=fmt, dpi=200)
