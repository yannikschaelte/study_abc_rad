import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import pickle

import slad
import pyabc

pyabc.settings.set_figure_params("pyabc")

distance_names = [
    "Adaptive__Euclidean__mad",
    "Adaptive__Manhattan__mad",
    "Adaptive__Manhattan__mad_or_cmad",
]

labels = [slad.C.distance_labels_short[dname] for dname in distance_names]
colors = [slad.C.distance_colors[dname] for dname in distance_names]

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))

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

    for h, label, color in zip(hs, labels, colors):
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
            ax=axes[i_problem],
        )
        axes[i_problem].set_title(slad.C.problem_labels[problem_type])

axes[0].set_title("Uninformative outlier")
axes[1].set_title("Conflicting replicate outliers")
axes[1].set_ylabel(None)
# axes[0].legend(bbox_to_anchor=[0.5, 0.5], loc="center")
axes[1].legend(bbox_to_anchor=[1, 0.5], loc="center")
fig.tight_layout()

for fmt in ["png", "pdf"]:
    plt.savefig(f"figures_robust/figure_motivation.{fmt}", format=fmt)
