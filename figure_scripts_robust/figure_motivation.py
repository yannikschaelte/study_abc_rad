import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import pickle

import slad
import pyabc

pyabc.settings.set_figure_params("pyabc")


n_sample = 30
alpha = 0.1


def load_data(problem, data_dir):
    data = {}
    for key in problem.get_y_keys():
        data[key] = np.loadtxt(os.path.join(data_dir, f"data_{key}.csv"), delimiter=",")
        if data[key].size == 1:
            data[key] = float(data[key])
    return data


def plot_sumstats(
    h, distance_name, label, arr_ax,
    show_distribution=True, show_mean=True,
):
    color = slad.C.distance_colors[distance_name]
    def f_plot(sum_stat, weight, arr_ax, **kwargs):
        for i, key in enumerate(sum_stat.keys()):
            arr_ax[i].plot(sum_stat[key], '-', color=color, alpha=alpha)

    def f_plot_mean(sum_stats, weights, arr_ax, **kwargs):
        aggregated = {}
        for key in sum_stats[0].keys():
            aggregated[key] = (np.array([sum_stat[key] for sum_stat in sum_stats]) \
                               * np.array(weights).reshape((-1,1))).sum(axis=0)
        for i, key in enumerate(aggregated.keys()):
            arr_ax[i].plot(aggregated[key], '-', color=color, alpha=1,
                           label=label)

    if not show_distribution:
        f_plot = None
    if not show_mean:
        f_plot_mean = None

    pyabc.visualization.plot_data_callback(
        h, f_plot, f_plot_mean, ax=arr_ax, n_sample=n_sample)


distance_names = [
    "Adaptive__Euclidean__mad",
    "Adaptive__Manhattan__mad",
    "Adaptive__Manhattan__mad_or_cmad",
]

labels = [slad.C.distance_labels_short[dname] for dname in distance_names]
labels[1] += " (new)"
labels[2] += " (new)"
colors = [slad.C.distance_colors[dname] for dname in distance_names]

fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(12, 3), constrained_layout=True)

fig = plt.figure(figsize=(12, 3))

gs1 = mpl.gridspec.GridSpec(1, 2)
gs1.update(left=0.06, right=0.47, top=0.79, bottom=0.3)
ax1 = plt.subplot(gs1[0])
ax2 = plt.subplot(gs1[1])

gs2 = mpl.gridspec.GridSpec(1, 2)
gs2.update(left=0.57, right=0.98, top=0.79, bottom=0.3)
ax3 = plt.subplot(gs2[0])
ax4 = plt.subplot(gs2[1])

axes = np.array([ax1, ax2, ax3, ax4])

for i_problem, (problem_type, problem, i_rep) in enumerate(zip(
    ["uninf", "gaussian"],
    [slad.UninfErrorProblem(), slad.GaussianErrorProblem()],
    [0, 0]
)):

    base_dir = os.path.join("data_hist", f"{problem.get_id()}_{i_rep}")

    hs = [
        pyabc.History("sqlite:///" + os.path.join(base_dir, f"db_{dname}.db"), create=False)
        for dname in distance_names
    ]

    # plot kdes
    for h, distance_name, label, color in zip(hs, distance_names, labels, colors):
        # plot kde
        pyabc.visualization.plot_kde_1d_highlevel(
            h,
            x="p0",
            xname=r"$\theta$",
            xmin=problem.get_prior_bounds()["p0"][0],
            xmax=problem.get_prior_bounds()["p0"][1],
            refval=problem.get_gt_par(),
            refval_color="grey",
            color=color,
            label=label,
            # bins=50,
            ax=axes[2 * i_problem],
        )
        # axes[i_problem].set_title(slad.C.problem_labels[problem_type])

    # plot distributions
    for h, distance_name, label in zip(hs, distance_names, labels):
        # plot fits
        plot_sumstats(
            h=h, arr_ax=[axes[2 * i_problem + 1]], show_mean=False,
            distance_name=distance_name, label=label,
        )

    # plot means
    for h, distance_name, label in zip(hs, distance_names, labels):
        plot_sumstats(
            h=h, arr_ax=[axes[2 * i_problem + 1]], show_distribution=False,
            distance_name=distance_name, label=label,
        )

    # plot data
    data_dir = os.path.join("data_robust", f"{problem.get_id()}_{i_rep}")
    data = load_data(problem, data_dir)
    for key in data.keys():
        axes[2 * i_problem + 1].plot(data[key], "x", color="black")
    axes[2 * i_problem + 1].set_ylabel("Value")

#fig.tight_layout(rect=(0, 0.1, 1, 0.8))

# labels
for ax in axes:
    ax.set_ylabel(None)
axes[0].set_title("ABC posterior")
axes[1].set_title("Data and simulations")
axes[2].set_title("ABC posterior")
axes[3].set_title("Data and simulations")
axes[1].set_xlabel("Coordinate")
axes[3].set_xlabel("Replicate")
fig.text(
    0.25, 0.98, "Model M1: Uninformative outlier",
    verticalalignment="top", horizontalalignment="center",
    fontsize=12)
fig.text(
    0.75, 0.98, "Model M2: Conflicting replicate outliers",
    verticalalignment="top", horizontalalignment="center",
    fontsize=12)

# legend
legend_elements = [
    mpl.lines.Line2D(
        [0], [0],
        color=color,
        label=label,
    )
    for color, label in zip(colors, labels)
]
legend_elements.append(mpl.lines.Line2D(
    [0], [0], linestyle="dotted", color="grey", label="True parameters"))
legend_elements.append(mpl.lines.Line2D(
    [0], [0], marker="x", color="black", linestyle="None", label="Observed data"))
axes[-1].legend(
    handles=legend_elements,
    loc="upper right",
    bbox_to_anchor=(1, -0.32),
    ncol=len(legend_elements),
)

for fmt in ["png", "pdf"]:
    plt.savefig(f"figures_robust/figure_motivation.{fmt}", format=fmt)
