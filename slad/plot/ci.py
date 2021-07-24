import matplotlib.pyplot as plt
import numpy as np
import pyabc


def plot_cis(problem, labels, iters=None, data_dir="data", problem_suff=""):
    if iters is None:
        iters = [None]
    bounds = problem.get_prior_bounds()
    pars = list(bounds.keys())

    n_label = len(labels)
    n_par = len(pars)

    fig, axes = plt.subplots(nrows=1, ncols=n_par, figsize=(4 * n_par, 4))
    if n_par == 1:
        axes = [axes]

    lbs, ubs = {}, {}
    for i_label, label in enumerate(labels):
        for iter in iters:
            if iter is None:
                h = pyabc.History(f"sqlite:///{data_dir}/{problem.get_id()}{problem_suff}/db_{label}.db", create=False)
            else:
                h = pyabc.History(f"sqlite:///{data_dir}/{problem.get_id()}{problem_suff}/db_{label}_{iter}.db", create=False)
            df, w = h.get_distribution()
            for i_par, par in enumerate(pars):
                lb = pyabc.weighted_quantile(df[par].to_numpy(), w, alpha=0.05)
                ub = pyabc.weighted_quantile(df[par].to_numpy(), w, alpha=0.95)

                lbs.setdefault(par, {}).setdefault(label, []).append(lb)
                ubs.setdefault(par, {}).setdefault(label, []).append(ub)

    for label in labels:
        for par in pars:
            lbs[par][label] = np.mean(lbs[par][label])
            ubs[par][label] = np.mean(ubs[par][label])

    for i_par, par in enumerate(pars):
        axes[i_par].set_xlim(bounds[par])

        for i_label, label in enumerate(labels):
            axes[i_par].hlines(
                y=n_label - i_label - 1, xmin=lbs[par][label], xmax=ubs[par][label]
            )

        axes[i_par].axvline(problem.get_gt_par()[par], color="grey")

        axes[i_par].set_yticks(range(n_label))
        axes[i_par].set_yticklabels(reversed(labels))
        axes[i_par].set_xlabel(par)

    fig.tight_layout()

    return fig, axes
