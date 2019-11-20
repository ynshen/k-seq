
from ..data.count_data import CountData
import cvxpy as cp
import numpy as np
import pandas as pd
import argparse
from yuning_util.dev_mode import DevMode
dev_mode = DevMode(pkg='k-seq')
dev_mode.on()


<<<<<<< HEAD:src/pkg/k_seq/estimator/count_model_mle_cvxpy.py
class CountData:
    """Class to store count data info for estimators

    Attributes:
        count (pd.DataFrame): dataframe contains the count info
        ctrl_vars (pd.DataFrame): dataframe stores controlled variable, e.g. BYO concentration
        input_pools (list): a list of sample name of input pools
        truth (pd.DataFrame): optional, store the true values of parameters for sequences in count table
        note (str): optional, any note about the dataset

    """

    def __init__(self, count, ctrl_vars, input_pools, truth=None, note=None):

        self.count = count
        self.ctrl_vars = ctrl_vars
        self.input_pools = input_pools
        self.truth = truth
        self.note = note

    @classmethod
    def from_simu_path(cls, path, input_pools=None, note=None):
        """Load data from a path of simulated data"""

        count = pd.read_csv(f'{path}/Y.csv', index_col='seq')
        ctrl_vars = pd.read_csv(f'{path}/x.csv', index_col=0)
        try:
            truth = pd.read_csv(f'{path}/truth.csv', index_col=0)
        except:
            truth = None
        if input_pools is None:
            input_pools = [sample for sample in ctrl_vars.columns
                           if np.isnan(ctrl_vars.loc['c', sample]) or ctrl_vars.loc['c', sample] < 0]
        return cls(count, ctrl_vars, input_pools, truth, note)

    @classmethod
    def from_SeqTable(cls, seq_table, table_name=None, input_pools=None, note=None):
        """Load data from a SeqTable object"""
        import pickle

        if isinstance(seq_table, str):
            with open(seq_table, 'rb') as handle:
                seq_table = pickle.load(handle)

        if table_name is None:
            table_name = 'table'
        count = getattr(seq_table, table_name)
        ctrl_vars = pd.DataFrame({'c': seq_table.x_values, 'n': count[seq_table.x_values].sum(axis=0)})
        ctrl_vars.index.name = 'params'
        if input_pools is None:
            input_pools = seq_table.grouper.input.group
        try:
            truth = getattr(seq_table, 'truth')
        except:
            truth = None

        return cls(count, ctrl_vars, input_pools, truth, note)

    def select(self, n, shuffle=False):
        from copy import deepcopy

        new_data = deepcopy(self)
        idx = np.arange(self.m)
        if shuffle:
            np.random.shuffle(idx)
        idx = idx[:n]
        new_data.count = self.count.iloc[idx, :]
        if hasattr(self, 'truth'):
            new_data.truth = self.truth.iloc[idx, :]
        new_data.note = f'{self.note} (select n={n})'

        return new_data

    @property
    def count_input(self):
        return self.count[self.input_pools]

    @property
    def ctrl_vars_input(self):
        return self.ctrl_vars[self.input_pools]

    @property
    def count_reacted(self):
        reacted_samples = [sample for sample in self.ctrl_vars.columns if sample not in self.input_pools]
        return self.count[reacted_samples]

    @property
    def ctrl_vars_reacted(self):
        reacted_samples = [sample for sample in self.ctrl_vars.columns if sample not in self.input_pools]
        return self.ctrl_vars[reacted_samples]

=======
>>>>>>> mle_tf:src/pkg/k_seq/estimator/count-model_mle_cvxpy.py
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


def main(output_dir, seq_table=None, table_name=None, input_pools=None,
         simu_data_path=None, solver='SCS', notes=None):


    from k_seq.utility.file_tools import dump_pickle
    import cvxpy as cp
    from pathlib import Path

    if seq_table is not None:
        count_data = CountData.from_SeqTable(seq_table=seq_table, table_name=table_name,
                                             input_pools=input_pools, note=notes)
    elif simu_data_path is not None:
        count_data = CountData.from_simu_path(path=simu_data_path, input_pools=input_pools, note=notes)

    prob, params = get_convex_model(count_data)
    print(f'Problem created, start solving with {solver}')

    if isinstance(solver, str):
        if solver.upper() == 'SCS':
            solver = cp.SCS
        if solver.upper() == 'MOSEK':
            solver = cp.MOSEK

    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True)

    run_solver(prob=prob, solver=solver, verbose=True, log_dir=output_dir + '/solver.log')
    print('Problem solving finished. Status unknown')
    dump_pickle(prob, path=output_dir + '/problem.pkl')
    dump_pickle(pd.DataFrame({'a': params.a.value, 'k': params.k.value, 'p0': params.p0.value}),
                path=output_dir + '/parameters.pkl')
    print(f'Running results saved to {output_dir}')


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run count model MLE using cvxpy')
    parser.add_argument('--output_path', '-o', type=str,
                        help='Path to save outputs')
    parser.add_argument('--seq_table', '-t', default=None,
                        help='path to pickled seqTable object')
    parser.add_argument('--table_name', '-n', default=None,
                        help='table name if use seq_table')
    parser.add_argument('--input_pools', '-i', default=None,
                        help="Name for input samples")
    parser.add_argument('--simu_data_path', '-s', default=None,
                        help='Path to the folder of simulated data')
    parser.add_argument('--solver', '-m', default='SCS',
                        help='Which solver to use, support SCS and MOSEK')
    parser.add_argument('--notes', default=None,
                        help='Optional notes for data')

    args = parser.parse_args()
    main(output_dir=args.output_path,
         seq_table=args.seq_table,
         table_name=args.table_name,
         input_pools=args.input_pools,
         simu_data_path=args.simu_data_path,
         solver=args.solver,
         notes=args.notes)
