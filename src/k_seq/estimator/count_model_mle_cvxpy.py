
import cvxpy as cp
import numpy as np


def get_convex_model(data):
    """Data is the CountData"""

    # Variables.
    seq_num, sample_num = data.count.shape
    p0 = cp.Variable((seq_num))
    a = cp.Variable((seq_num))
    k = cp.Variable((seq_num))
    t = 90
    alpha = 0.476

    # LL matrix
    # LL for initial pool
    # n_ij \log p_i0 for any j in initial pool
    ll_init = cp.sum(
        cp.multiply(data.count_input.values.T,
                    cp.log(cp.vstack([p0 for _ in range(data.count_input.shape[1])])))
    )
    ## LL for reacted pool
    # n_ij (\log p_i0 + \log a_i + \log (1 - exp(-alpha * t * k_i * c_js )))
    ll_reacted = cp.sum(
        cp.multiply(
            data.count_reacted.values.T,
            cp.vstack([(cp.log(p0) + cp.log(a)) for _ in range(data.ctrl_vars_reacted.shape[1])]) \
            + cp.log(1 - cp.exp(- alpha * t * cp.vstack([c * k for c in data.ctrl_vars_reacted.loc['c'].values])))
        )
    )

    # Convex neg-likelihood function
    nll = -ll_init - ll_reacted
    objective = cp.Minimize(nll)

    # Constraints.
    constraints = (
        [0 <= p0,
         cp.sum(p0) == 1,
         a >= 0,
         a <= 1,
         k >= 0]
    )

    # Problem.
    prob = cp.Problem(objective, constraints)

    from collections import namedtuple
    Params = namedtuple(field_names=['p0', 'a', 'k'], typename='params')

    return prob, Params(p0=p0, a=a, k=k)


def run_solver(prob, solver=cp.SCS, verbose=True, log_dir=None):
    from k_seq.utility.log import Timer, FileLogger

    if log_dir is not None:
        with FileLogger(file_path=log_dir), Timer():
            solution = prob.solve(solver, verbose=verbose)
    else:
        with Timer():
            solution = prob.solve(solver, verbose=verbose)


def compare_to_truth(data, params, **kwargs):
    import matplotlib.pyplot as plt
    plt.style.use('seaborn')
    fig, axes = plt.subplots(1, 4, figsize=[16, 4])

    def scale_free_plot(x, y, ax, log=False, label=None, **kwargs):
        from scipy import stats

        x_lim = np.min(x) * 0.9, np.max(x) * 1.1
        y_lim = np.min(y) * 0.9, np.max(y) * 1.1
        ax.scatter(x, y, **kwargs)
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        if log:
            ax.set_xscale('log')
            ax.set_yscale('log')
        ax.set_title(f"Pearson: {stats.pearsonr(x, y)[0]:.4f}\nSpearman: {stats.spearmanr(x, y)[0]:.4f}")
        ax.tick_params(which='both', bottom=False, labelbottom=False, left=False, labelleft=False)
        if label is not None:
            ax.set_xlabel(label, fontsize=12)

    scale_free_plot(x=data.truth['p0'], y=params.p0.value, ax=axes[0], log=True, label='p0', **kwargs)
    scale_free_plot(x=data.truth['a'], y=params.a.value, ax=axes[1], log=False, label='a', **kwargs)
    scale_free_plot(x=data.truth['k'], y=params.k.value, ax=axes[2], log=True, label='k' ,**kwargs)
    scale_free_plot(x=data.truth['k'] * data.truth['a'],
                    y=params.a.value * params.k.value,
                    ax=axes[3], log=True, label='a*k', **kwargs)
    plt.show()


