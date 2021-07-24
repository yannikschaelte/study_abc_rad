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

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 6), )

for deviation in [0, 1, 5, 25, 125]:
    for i_problem, (problem_type, problem) in enumerate([
        ("uninf", slad.UninfErrorProblem()),
        ("gaussian", slad.GaussianErrorProblem()),
    ]):
        for i_distance, distance_name in enumerate(distance_names):
            h = pyabc.History(
                f"sqlite:///data_deviation/{problem.get_id()}_{deviation}/db_{distance_name}.db", create=False)

            pyabc.visualization.plot_kde_1d_highlevel(
                h,
                x="p0",
                xmin=problem.get_prior_bounds()["p0"][0],
                xmax=problem.get_prior_bounds()["p0"][1],
                xname=slad.C.parameter_labels[problem_type]["p0"],
                ax=axes[i_problem, i_distance],
                refval=problem.get_gt_par(),
                refval_color="grey",
                numx=1000,
                label=f"Deviation {deviation} x std",
            )
            axes[i_problem, i_distance].set_title(
                f"Model {slad.C.problem_labels[problem_type]} - {slad.C.distance_labels_short[distance_name]}")
            if i_distance > 0:
                axes[i_problem, i_distance].set_ylabel(None)


# axes[0, 2].legend()
for ax in [axes[0, 0,], axes[1, 0]]:
    ax.set_ylabel("ABC posterior")
axes[0, 2].legend(bbox_to_anchor=(0.9, 0.5), loc="center")
fig.tight_layout(h_pad=3)

for fmt in ["png", "pdf"]:
    plt.savefig(f"figures_robust/figure_deviation.{fmt}", format=fmt)
