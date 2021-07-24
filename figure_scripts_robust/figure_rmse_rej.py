import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import pickle

import slad
import pyabc

pyabc.settings.set_figure_params("pyabc")


n_rep = 20

fontsize = 10
fontsize_small = 8
padding = 0.3

distance_names = [
    "Adaptive__Euclidean__mad",
    "Adaptive__Manhattan__mad_or_cmad",
]

def create_vals(problem_type):
    pickle_file = f"figures_robust/data_{problem_type}_rej.pickle"
    if os.path.exists(pickle_file):
        with open(pickle_file, "rb") as f:
            means, stds, gt_par = pickle.load(f)
            return means, stds, gt_par

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
    n_dist = len(distance_names)

    vals = np.full(shape=(n_dist, 2, n_par, n_rep), fill_value=np.nan)

    for i_mode, rej in enumerate([False, True]):
        kwargs = {"n_obs_error": 0}
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

        for i_dist, distance_name in enumerate(distance_names):
            for i_rep in range(n_rep):
                if rej:
                    data_dir = "data_robust"
                else:
                    data_dir = "data_rej"

                h = pyabc.History(
                    f"sqlite:///{data_dir}/{problem.get_id()}_{i_rep}/db_{distance_name}.db",
                    create=False)

                df, w = h.get_distribution(t=h.max_t)
                vals[i_dist, i_mode, :, i_rep] = np.array(
                    [pyabc.weighted_rmse(df[key], w, gt_par[key])
                     for key in gt_par])

                # print(vals[i_dist, i_mode, :, i_rep])

    means = np.mean(vals, axis=3)
    stds = np.std(vals, axis=3)

    with open(pickle_file, "wb") as f:
        pickle.dump((means, stds, gt_par), f)

    return means, stds, gt_par


def plot_rmse(problem_type, log: bool, fig, ylabels: bool, width: float):
    print(problem_type)
    means, stds, gt_par = create_vals(problem_type)
    n_par = len(gt_par)
    axes = fig.subplots(nrows=1, ncols=n_par)
    if n_par == 1:
        axes = [axes]

    n_dist = len(distance_names)
    colors = [slad.C.distance_colors[dname] for dname in distance_names]
    for i_par, key in enumerate(gt_par.keys()):
        ax = axes[i_par]
        ys = np.arange(n_dist)
        if ylabels and i_par == 0:
            ax.set_yticks(np.arange(n_dist))
            ax.set_yticklabels([
                slad.C.distance_labels_short[dname] for dname in distance_names],
                fontdict={"fontsize": fontsize_small})
            ax.xaxis.set_ticks_position("none")
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
                ax.text(max_val * 0.99,
                        i_dist - (-1)**i * 0.2,
                        f"{means[i_dist, i, i_par]:.3f}",
                        fontdict={"fontsize": fontsize_small},
                        verticalalignment="center",
                        horizontalalignment="right")

        #ax.set_xlabel("RMSE")
        ax.set_title(slad.C.parameter_labels[problem_type][key], fontsize=fontsize)
        #ax.axhline(y=3.5, color="grey", linestyle="--")

        plt.setp(ax.get_xticklabels(), fontsize=fontsize)

    # fig.suptitle(problem_labels[problem_type])
    #fig.tight_layout()
    fig.suptitle(slad.C.problem_labels[problem_type], x=0.5)
    fig.subplots_adjust(left=padding / width, right=1 - padding / width, top=0.8)
    #plt.savefig(f"plot_robust/rmse_{problem_type}.png")

fig = plt.figure(figsize=(14, 2))
width_ratios = [2, 2, 4, 8, 6]
subfigs = fig.subfigures(nrows=1, ncols=5, wspace=0.01, width_ratios=width_ratios)

for i, problem_type in enumerate(["gaussian", "cr-zero", "gk", "lv"]):
    plot_rmse(problem_type, log=False, fig=subfigs[i+1], ylabels=False,
              width = width_ratios[i+1])

#subfigs[0].subplots_adjust(left=padding / 4 / width_ratios[0], right=1 - 0.4 / width_ratios[0])
subfigs[-1].subplots_adjust(left=padding / width_ratios[-1],
                            right=1 - padding / 4 / width_ratios[-1])

ax = subfigs[0].subplots()
for i_dist, distance_name in enumerate(reversed(distance_names)):
    ax.text(0, 0.2+0.99*(i_dist / len(distance_names)),
            slad.C.distance_labels_short[distance_name],
            transform=ax.transAxes,
            fontsize=fontsize)
ax.set_axis_off()
subfigs[0].subplots_adjust(left=padding / 4 / width_ratios[0], right=1 - padding / 4 / width_ratios[0])


#fig, axes = plt.subplots(
#    nrows=1, ncols=13, figsize=(14, 4),
#    gridspec_kw={"width_ratios": [6, 0.05, 5, 5, 0.05, 5, 5, 5, 5, 0.05, 5, 5, 5]})

#axes = axes.flatten()

#plot_rmse("gaussian", True, [axes[0]], ylabels=True)
#plot_rmse("cr-zero", True, axes[[2, 3]])
#plot_rmse("gk", True, axes[[5, 6, 7, 8]])
#plot_rmse("lv", True, axes[[10, 11, 12]])

#for ix in [1, 4, 9]:
#    axes[ix].axis("off")

#fig.tight_layout()

# fig.suptitle("RMSE")

for fmt in ["svg", "png"]:
    plt.savefig(f"figures_robust/figure_rmse_rej.{fmt}", format=fmt, dpi=200)
