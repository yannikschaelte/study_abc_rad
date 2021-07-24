import os
import numpy as np
import matplotlib.pyplot as plt

import slad


def load_data(problem, data_dir):
    data = {}
    for key in problem.get_y_keys():
        data[key] = np.loadtxt(os.path.join(data_dir, f"data_{key}.csv"), delimiter=",")
        if data[key].size == 1:
            data[key] = float(data[key])
    return data


dir = os.path.dirname(os.path.realpath(__file__))
n_rep = 20

for problem_class in [
    slad.PrangleLVErrorProblem,
    slad.PrangleGKErrorProblem,
    slad.CRErrorZeroProblem,
    # slad.CRErrorSwapProblem,
    slad.GaussianErrorProblem,
    slad.UninfErrorProblem,
]:
    for kwargs in [{"n_obs_error": 0}, {}]:
        problem = problem_class(**kwargs)
        nrows = int(np.sqrt(n_rep))
        ncols = int(np.ceil(n_rep / nrows))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 4 * nrows))
        for i_rep in range(n_rep):
            data_dir = os.path.join(dir, "..", "data_robust", f"{problem.get_id()}_{i_rep}")
            data = load_data(problem, data_dir)
            ax = axes.flatten()[i_rep]
            if isinstance(problem, slad.PrangleLVErrorProblem):
                ax.plot(data["y"][:, 0], "x-")
                ax.plot(data["y"][:, 1], "x-")
            else:
                ax.plot(data["y"], "x-")
            ax.set_title(f"Replicate {i_rep}")
        for i_rep in range(n_rep, nrows * ncols):
            axes.flatten()[i_rep].set_axis_off()

        plt.savefig(f"plot_robust/data/data_{problem.get_id()}.png")
