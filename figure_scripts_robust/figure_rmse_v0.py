import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

import slad
import pyabc

pyabc.settings.set_figure_params("pyabc")


n_rep = 20

fontsize = 10
fontsize_small = 8


def create_vals(problem_type):
    if problem_type == "gaussian":
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
    n_dist = len(slad.C.distance_names)

    vals = np.full(shape=(n_dist, 2, n_par, n_rep), fill_value=np.nan)

    for i_mode, kwargs in enumerate([{'n_obs_error': 0}, {}]):
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

        for i_dist, distance_name in enumerate(slad.C.distance_names):
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

    return means, stds, gt_par


def plot_rmse(problem_type, log: bool, axes, ylabels: bool = False):
    print(problem_type)
    means, stds, gt_par = create_vals(problem_type)

    n_dist = len(slad.C.distance_names)
    colors = list(slad.C.distance_colors.values())
    for i_par, key in enumerate(gt_par.keys()):
        ax = axes[i_par]
        ys = np.arange(n_dist)
        if ylabels and i_par == 0:
            ax.set_yticks(np.arange(n_dist))
            ax.set_yticklabels([
                slad.C.distance_labels_short[dname] for dname in slad.C.distance_names],
                fontdict={"fontsize": fontsize_small})

        else:
            ax.set_yticks([])
        ax.invert_yaxis()
        ax.barh(
            ys - 0.2, means[:, 0, i_par],
            #xerr=stds[:, 0, i_par],
            color=colors, alpha=0.3, height=0.4,
            error_kw={"ecolor": "grey"},
        )
        ax.barh(
            ys + 0.2, means[:, 1, i_par],
            #xerr=stds[:, 0, i_par],
            color=colors, alpha=0.8, height=0.4,
            error_kw={"ecolor": "grey"},
        )
        if log:
            ax.set_xscale("log")

        # add value
        for i_dist in range(n_dist):
            for i in [0, 1]:
                max_val = means[:, :, i_par].max()
                if log:
                    pos_x = means[i_dist, i, i_par] * (1 + 1 / max_val)
                else:
                    pos_x = means[i_dist, i, i_par] + (1  + 1 / max_val)
                ax.text(max_val * 0.9,
                        i_dist - (-1)**i * 0.2,
                        f"{means[i_dist, i, i_par]:.3f}",
                        fontdict={"fontsize": fontsize_small},
                        verticalalignment="center",
                        horizontalalignment="right")

        #ax.set_xlabel("RMSE")
        ax.set_title(key, fontsize=fontsize)
        ax.axhline(y=3.5, color="grey", linestyle="--")

        plt.setp(ax.get_xticklabels(), fontsize=fontsize)

    # fig.suptitle(problem_labels[problem_type])
    fig.tight_layout()
    plt.savefig(f"plot_robust/rmse_{problem_type}.png")

fig, axes = plt.subplots(
    nrows=1, ncols=13, figsize=(14, 4),
    gridspec_kw={"width_ratios": [6, 0.05, 5, 5, 0.05, 5, 5, 5, 5, 0.05, 5, 5, 5]})

axes = axes.flatten()

plot_rmse("gaussian", True, [axes[0]], ylabels=True)
plot_rmse("cr-zero", True, axes[[2, 3]])
plot_rmse("gk", True, axes[[5, 6, 7, 8]])
plot_rmse("lv", True, axes[[10, 11, 12]])

for ix in [1, 4, 9]:
    axes[ix].axis("off")

fig.tight_layout()

plt.savefig("figures_robust/figure_rmse_base.svg", format="svg")
