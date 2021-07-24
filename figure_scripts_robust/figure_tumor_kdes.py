import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse

import slad
import pyabc

parser = argparse.ArgumentParser()
parser.add_argument("--frac_error", type=float)
parser.add_argument("--hist", type=int)
args = parser.parse_args()
use_hist = args.hist
if use_hist == 1:
    db_dir_base = "data_hist"
    hist_suf = "_hist"
elif use_hist == 0:
    db_dir_base = "data_robust"
    hist_suf = ""

pyabc.settings.set_figure_params("pyabc")

distance_names = [
    "Calibrated__Euclidean__mad",
    "Adaptive__Euclidean__mad",
    "Adaptive__Manhattan__mad",
    "Adaptive__Manhattan__mad_or_cmad",
]

problem = slad.TumorErrorProblem(noisy=True, frac_error=0)
problem_error = slad.TumorErrorProblem(noisy=True, frac_error=0.1)
bounds = problem.get_prior_bounds()
refval = problem.get_gt_par()

pars = list(bounds.keys())

reduced = True
if reduced:
    pars = [
        "log_division_rate",
        "log_division_depth",
        "log_initial_spheroid_radius",
        "log_ecm_production_rate",
        "log_ecm_degradation_rate",
    ]

fig = plt.figure(
    figsize=(12, 5),
    #constrained_layout=True,
)
arr_ax = fig.subplots(
    nrows=2,
    ncols=len(pars),
    #gridspec_kw={"hspace": 0.15},
)

for distance_name in distance_names:
    h = pyabc.History(
        f"sqlite:///{db_dir_base}/{problem.get_id()}_0/db_{distance_name}.db",
        create=False)
    df, w = h.get_distribution()
    for i_par, par in enumerate(pars):
        pyabc.visualization.plot_kde_1d(
            df, w, x=par, xmin=bounds[par][0], xmax=bounds[par][1],
            refval=refval, refval_color="grey", ax=arr_ax[0, i_par],
            color=slad.C.distance_colors[distance_name],
            label=slad.C.distance_labels_short[distance_name],
            xname=slad.C.parameter_labels["tumor"][par],
            numx=500,
        )

    h = pyabc.History(
        f"sqlite:///{db_dir_base}/{problem_error.get_id()}_0/db_{distance_name}.db",
        create=False)
    df, w = h.get_distribution()
    for i_par, par in enumerate(pars):
        pyabc.visualization.plot_kde_1d(
            df, w, x=par, xmin=bounds[par][0], xmax=bounds[par][1],
            refval=refval, refval_color="grey", ax=arr_ax[1, i_par],
            color=slad.C.distance_colors[distance_name],
            label=slad.C.distance_labels_short[distance_name],
            xname=slad.C.parameter_labels["tumor"][par],
            numx=500,
        )

for ax in arr_ax[:, 1:].flatten():
    ax.set_ylabel(None)
arr_ax[0, 0].set_ylabel("ABC posterior")
arr_ax[1, 0].set_ylabel("ABC posterior")
for ax in arr_ax.flatten():
    ax.set_xlabel(None)
for ax, par in zip(arr_ax[0, :], pars):
    ax.set_title(slad.C.parameter_labels["tumor"][par], fontsize=10)

fig.tight_layout(rect=(0.01, 0.05, 0.99, 0.95), h_pad=2) # left, bottom, right, top

fig.text(
   0.5, 0.97, "M6, Outlier-free data",
   horizontalalignment="center", verticalalignment="top", fontsize=12)
fig.text(
   0.5, 0.49, "M6, Outlier-corrupted data",
   horizontalalignment="center", verticalalignment="top", fontsize=12)

# legend
legend_elements = [
    mpl.lines.Line2D(
        [0], [0],
        color=slad.C.distance_colors[dname],
        label=slad.C.distance_labels_short[dname],
    )
    for dname in distance_names
]
arr_ax.flatten()[-1].legend(
    handles=legend_elements,
    loc="upper right",
    bbox_to_anchor=(1, -0.16),
    ncol=len(legend_elements),
)

for fmt in ["pdf", "png"]:
    plt.savefig(f"figures_robust/figure_tumor_kdes{hist_suf}.{fmt}", format=fmt, dpi=200)
