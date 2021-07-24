import matplotlib.pyplot as plt
import numpy as np
import pyabc


def plot_1d_kdes(problem, labels, iters=None, data_dir="data", problem_suff=""):
    if iters is None:
        iters = [None]
    bounds = problem.get_prior_bounds()
    pars = list(bounds.keys())
    refval = problem.get_gt_par()

    n_label = len(labels)
    n_par = len(pars)

    fig, axes = plt.subplots(
        nrows=n_label, ncols=n_par, figsize=(3 * n_par, 3 * n_label)
    )
    axes = axes.reshape(n_label, n_par)

    for i_label, label in enumerate(labels):
        for iter in iters:
            if iter is None:
                h = pyabc.History(f"sqlite:///{data_dir}/{problem.get_id()}{problem_suff}/db_{label}.db", create=False)
            else:
                h = pyabc.History(f"sqlite:///{data_dir}/{problem.get_id()}{problem_suff}/db_{label}_{iter}.db", create=False)
            df, w = h.get_distribution()
            for i_par, par in enumerate(pars):
                pyabc.visualization.plot_kde_1d(
                    df,
                    w,
                    x=par,
                    xmin=bounds[par][0],
                    xmax=bounds[par][1],
                    refval=refval,
                    refval_color="grey",
                    ax=axes[i_label, i_par],
                )
        for i_par in range(n_par):
            if i_par == 0:
                axes[i_label, i_par].set_ylabel(label)
            else:
                axes[i_label, i_par].set_ylabel(None)
            if i_label < n_label - 1:
                axes[i_label, i_par].set_xlabel(None)

    fig.tight_layout()

    return fig, axes


def plot_1d_kdes_integrated(problem, labels, data_dir="data", problem_suff="", db_suff="", pretty_labels=None):
    if pretty_labels is None:
        pretty_labels = {key: key for key in labels}
    bounds = problem.get_prior_bounds()
    pars = list(bounds.keys())
    refval = problem.get_gt_par()

    n_label = len(labels)
    n_par = len(pars)

    ncol = 4
    nrow = int(np.ceil(n_par / ncol))
    print(ncol, nrow)
    fig, axes = plt.subplots(
        nrows=nrow, ncols=ncol, figsize=(3*ncol, 3*nrow)
    )

    for i_label, label in enumerate(labels):
        h = pyabc.History(f"sqlite:///{data_dir}/{problem.get_id()}{problem_suff}/db_{label}{db_suff}.db", create=False)
        df, w = h.get_distribution()
        for i_par, par in enumerate(pars):
            pyabc.visualization.plot_kde_1d(
                df,
                w,
                x=par,
                xmin=bounds[par][0],
                xmax=bounds[par][1],
                refval=refval,
                refval_color="grey",
                ax=axes.flatten()[i_par],
                color=f"C{i_label}",
                label=pretty_labels[label],
            )
    for i_par in range(n_par):
        if i_par > 0:
            axes.flatten()[i_par].set_ylabel(None)
    for i_par in range(n_par, len(axes.flatten())):
        axes.flatten()[i_par].axis("off")

    fig.tight_layout()
    axes.flatten()[n_par-1].legend(loc="upper left", bbox_to_anchor=(1, 0.9))

    return fig, axes
