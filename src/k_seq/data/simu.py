"""Module contains code to simulate data"""
import numpy as np
import pandas as pd

from ..utility import DocHelper
from ..utility.func_tools import is_numeric
from yutility import logging


class DistGenerators:
    """A collection of random value generators from commonly used distributions.
    `uniq_seq_num` number of independent draw of distribution are returned

    Available distributions:
        lognormal
        uniform
        compo_lognormal
    """

    def __init__(self):
        pass

    @staticmethod
    def lognormal(size=None, loc=None, scale=None, c95=None, seed=None):
        """Sample from a log-normal distribution
        indicate with `loc` and `scale`, or `c95`

        Args:
            size (`int`): number of values to draw
            loc (`float`): center of log-normal distribution, default 0
            scale (`float`): log variance of the distribution, default 0
            c95 ([`float`, `float`]): 95% percentile of log-normal distribution
            seed: random seed

        Returns:
            a draw from distribution with given uniq_seq_num
        """

        if c95 is None:
            if loc is None:
                loc = 0
            if scale is None:
                scale = 0
        else:
            c95 = np.log(np.array(c95))
            loc = (c95[0] + c95[1]) / 2
            scale = (c95[1] - c95[0]) / 3.92

        if seed is not None:
            np.random.seed(seed)

        if size is None:
            return np.exp(np.random.normal(loc=loc, scale=scale))
        else:
            return np.exp(np.random.normal(loc=loc, scale=scale, size=size))

    @staticmethod
    def uniform(low=None, high=None, size=None, seed=None):
        """Sample from a uniform distribution"""

        if seed is not None:
            np.random.seed(seed)
        if size is None:
            return np.random.uniform(low=low, high=high)
        else:
            return np.random.uniform(low=low, high=high, size=size)

    @staticmethod
    def compo_lognormal(size, loc=None, scale=None, c95=None, seed=None):
        """Sample a pool composition from a log-normal distribution
        indicate with `loc` and `scale`, or `c95`

        Example:
            scale = 0 means an evenly distributed pool with all components have relative abundance 1/uniq_seq_num

        Args:
            size (`int`): uniq_seq_num of the pool
            loc (`float`): center of log-normal distribution
            scale (`float`): log variance of the distribution
            c95 ([`float`, `float`]): 95% percentile of log-normal distribution
            seed: random seed

        """

        if c95 is None:
            if loc is None or scale is None:
                raise ValueError('Please indicate loc/scale or c95')
        else:
            c95 = np.log(np.array(c95))
            loc = (c95[0] + c95[1]) / 2
            scale = (c95[1] - c95[0]) / 3.92
        if seed is not None:
            np.random.seed(seed)

        q = np.exp(np.random.normal(loc=loc, scale=scale, size=size))
        return q / np.sum(q)


simu_doc = DocHelper(
    uniq_seq_num=('int', 'Number of unique sequences from simulation'),
    p0=('list-like, generator, or callable', 'reserved argument for initial pool composition (fraction)'),
    depth=('int or float', 'sequence depth defined on mean reads per sequence'),
    seed=('int', 'random seed for repeatability'),
    param_generators=('kwargs of list-like, generator, or callable', 'parameter generator depending on the model'),
    x_values=('list-like', 'list of controlled variables in each experiment setup,'
                           'negative value means it is an initial pool'),
    sample_reads=('int or list-like', 'Number of total reads for each sample'),
    kinetic_model=('callable', 'model the amount of sequences in reaction given input variables. '
                               'Default `BYOModel.amount_first_order`'),
    count_model=('callable', 'model the sequencing counts w.r.t. total reads and pool composition.'
                             'Default MultiNomial model.'),
    total_amount_error=('float or callable', 'float as the standard deviation for a fixed Gaussian error, '
                                             'or any error function on the DNA amount. Use 0 for no introduced error.'),
    replace=('bool', 'if sample with replacement when sampling from a dataframe'),
    reps=('int', 'number of replicates for each condition in x_values'),
    save_to=('str', 'optional, path to save the simulation results with x, Y, truth csv file '
                    'and a pickled SeqData object'),
    x=('pd.DataFrame', 'controlled variable (c, n) for samples'),
    Y=('pd.DataFrame', 'simulated sequence counts for given samples'),
    param_table=('pd.DataFrame', 'seq_table list the parameters for simulated sequences, including p0, k, A, kA'),
    seq_table=('data.SeqData', 'a SeqData object to stores all the data'),
    truth=('pd.DataFrame', 'true values of parameters (e.g. p0, k, A) for simulated sequences')
)

accepted_gen_type = """
Accepted parameter input:
  - list-like: if uniq_seq_num does not match as expected uniq_seq_num, resample with replacement
  - generator: given generator returns a random parameter
  - callable: if taking `uniq_seq_num` as an argument, needs to return a uniq_seq_num vector of sampled parameter or a
    generator to generate a uniq_seq_num vector; if not taking `uniq_seq_num` as an argument, needs to return single 
    sample"""


class PoolParamGenerator:
    """Functions to generate parameters for a set of sequence in a sequence pool

    Methods:
        - sample_from_iid_dist
        - sample_from_dataframe

    Returns:
        pd.DataFrame with columns of parameters and rows of simulated parameter for each sequence
    """

    @staticmethod
    @simu_doc.compose(f"""Simulate the seq parameters from individual draws of distributions
Parameter:
    p0: initial fraction for each sequence for uneven pool
    depending on the model. e.g. first-order model needs to include k and A

    {accepted_gen_type}
    
Args:
<< p0, uniq_seq_num, seed, param_generators>>
    
Returns:
    a n_row = uniq_seq_num pd.DataFrame contains generated parameters
""")
    def sample_from_iid_dist(uniq_seq_num, seed=None, **param_generators):

        def generate_params(param_input):
            """Parse single distribution input and reformat as generated results
            """

            from types import GeneratorType

            if isinstance(param_input, (list, np.ndarray, pd.Series)):
                if len(param_input) == uniq_seq_num:
                    return param_input
                else:
                    logging.info('Size fo input param list and expected uniq_seq_num does not match, '
                                 'resample to given uniq_seq_num with replacement')
                    return np.random.choice(param_input, replace=True, size=uniq_seq_num)
            elif isinstance(param_input, GeneratorType):
                # assume only generate one realization
                return [next(param_input) for _ in range(uniq_seq_num)]
            elif callable(param_input):
                try:
                    # if there is a uniq_seq_num parameter to pass
                    param_output = param_input(size=uniq_seq_num)
                    if isinstance(param_output, (list, np.ndarray, pd.Series)):
                        return param_output
                    elif isinstance(param_output, GeneratorType):
                        return next(param_output)
                    else:
                        logging.error("Unknown input to draw a distribution value", error_type=TypeError)
                except TypeError:
                    # if can not pass uniq_seq_num, assume generate single samples
                    param_output = param_input()
                    if isinstance(param_output, GeneratorType):
                        return [next(param_output) for _ in range(uniq_seq_num)]
                    elif isinstance(param_output, (float, int)):
                        return [param_input() for _ in range(uniq_seq_num)]
                    else:
                        logging.error("Unknown callable return type for distribution",
                                      error_type=TypeError)
            else:
                logging.error("Unknown input to draw a distribution value", error_type=TypeError)

        if seed is not None:
            np.random.seed(seed)

        results = pd.DataFrame(data={**{param: list(generate_params(gen)) for param, gen in param_generators.items()}})

        if 'p0' in results.columns:
            results['p0'] = results['p0'] / results['p0'].sum()
        else:
            results['p0'] = np.repeat(1 / uniq_seq_num, repeats=uniq_seq_num)

        return results

    @classmethod
    @simu_doc.compose("""Simulate parameter by resampling rows of a given dataframe
Args:
    df (pd.DataFrame): dataframe contains parameters as columns to sample from,
      needs to have `p0` as one column for heterogenous pool
<<uniq_seq_num>>
""")
    def sample_from_dataframe(cls, df, uniq_seq_num, replace=True, weights=None, seed=None):

        results = df.sample(n=int(uniq_seq_num), replace=replace, weights=weights, random_state=seed)
        if 'p0' in results.columns:
            results['p0'] = results['p0'] / results['p0'].sum()
        else:
            results['p0'] = np.repeat(1 / uniq_seq_num, repeats=uniq_seq_num)
        results.index = np.arange(results.shape[0])

        return results


@simu_doc.compose("""Simulate sequencing count dataset given kinetic and count model

Procedure:
  1. parameter for each unique sequences were sampled from param_sample_from_df and kwargs
    (param_generators). It is an even pool if p0 is not provided. No repeated parameters.
  2. simulate the reacted amount / fraction of sequences with each controlled variable in x_values
  3. Simulated counts with given total total_reads were simulated for input pool and reacted pools.

Args:
    <<uniq_seq_num, x_values, total_reads, p0, kinetic_model, count_model>>
    param_sample_from_df (pd.DataFrame): optional to sample sequences from given table
    weights (list or str): weights/col of weight for sampling from table
    <<total_amount_error, conv_reps, seed, save_to, param_generator>>

Returns:
    x (pd.DataFrame): c, n value for samples
    Y (pd.DataFrame): simulated sequence counts for given samples
    param_table (pd.DataFrame): seq_table list the parameters for simulated sequences
    seq_table (data.SeqData): a SeqData object to stores all the data
""")
def simulate_counts(uniq_seq_num, x_values, total_reads, p0_generator=None,
                    kinetic_model=None, count_model=None, total_amount_error=None,
                    param_sample_from_df=None, weights=None, replace=True,
                    reps=1, seed=None, note=None,
                    save_to=None, **param_generators):
    from ..model import pool

    if seed is not None:
        np.random.seed(seed)

    # default models
    # kinetic_model: BYO first-order model returns absolute amount
    # count_model: multinomial
    if kinetic_model is None:
        from ..model import kinetic
        kinetic_model = kinetic.BYOModel.amount_first_order(broadcast=False)
        logging.info('No kinetic model provided, use BYOModel.amount_first_order')
    if count_model is None:
        from ..model import count
        count_model = count.multinomial
        logging.info('No count model provided, use multinomial distribution')

    # compose sequence parameter from (with priority high to low)
    # 1. p0
    # 2. param_generator
    # 3. param_sample_from_df

    param_table = pd.DataFrame(index=np.arange(uniq_seq_num))
    logging.info(f'param_table created, param_table shape {param_table.shape}')
    # if sample p0 from a generator
    if p0_generator is not None:
        param_table['p0'] = PoolParamGenerator.sample_from_iid_dist(p0=p0_generator,
                                                                    uniq_seq_num=uniq_seq_num)['p0']
        logging.info(f'p0 added from distribution, param_table shape {param_table.shape}')
    # if extra param_generator detected
    if param_generators != {}:
        temp_table = PoolParamGenerator.sample_from_iid_dist(uniq_seq_num=uniq_seq_num,
                                                             **param_generators)
        col_name = temp_table.columns[~temp_table.columns.isin(param_table.columns.values)]
        param_table = pd.concat([param_table, temp_table[col_name]], ignore_index=True, axis=1)
        logging.info(f'{list(param_generators.keys())} added from distribution, param_table shape {param_table.shape}')

    # if a param dataframe if provided
    if param_sample_from_df is not None:
        temp_table = PoolParamGenerator.sample_from_dataframe(
            df=param_sample_from_df,
            uniq_seq_num=uniq_seq_num,
            replace=replace,
            weights=weights
        )
        col_name = temp_table.columns[~temp_table.columns.isin(param_table.columns.values)]
        param_table = pd.concat([param_table, temp_table[col_name]], axis=1)
        logging.info(f'{col_name} added from dataframe, param_table shape {param_table.shape}')
    param_table.index.name = 'seq'

    # get pool model
    pool_model = pool.PoolModel(count_model=count_model,
                                kinetic_model=kinetic_model,
                                param_table=param_table)
    x = {}
    Y = {}
    total_amount = {}

    if is_numeric(total_reads):
        total_reads = np.repeat(total_reads, len(x_values))
    for sample_ix, (c, n) in enumerate(zip(x_values, total_reads)):
        if reps is None or reps == 1:
            total_amount[f"s{sample_ix}"], Y[f"s{sample_ix}"] = pool_model.predict(c=c, N=n)
            x[f"s{sample_ix}"] = {'c': c, 'n': n}
        else:
            for rep in range(reps):
                total_amount[f"s{sample_ix}-{rep}"], Y[f"s{sample_ix}-{rep}"] = pool_model.predict(c=c, N=n)
                x[f"s{sample_ix}-{rep}"] = {'c': c, 'N': n}
    # return x, Y, total_amounts, param_table
    x = pd.DataFrame.from_dict(x, orient='columns')
    Y = pd.DataFrame.from_dict(Y, orient='columns')
    total_amount = pd.Series(total_amount)

    if total_amount_error is not None:
        if is_numeric(total_amount_error):
            total_amount += np.random.normal(loc=0, scale=total_amount_error, size=len(total_amount))
        elif callable(total_amount_error):
            total_amount = total_amount.apply(total_amount_error)
        else:
            logging.error('Unknown total_amount_error type', error_type=TypeError)

    x.index.name = 'param'
    Y.index.name = 'seq'
    total_amount.index.name = 'amount'

    # return x, Y, total_amount, param_table, None
    from .seq_data import SeqData

    input_samples = x.loc['c']
    input_samples = list(input_samples[input_samples < 0].index)

    seq_table = SeqData(data=Y, x_values=x.loc['c'], note=note, data_unit='counts',
                        grouper={'input': input_samples,
                                 'reacted': [sample for sample in x.columns if sample not in input_samples]})
    seq_table.add_sample_total(total_amounts=total_amount.to_dict(),
                               full_table=seq_table.table.original)
    seq_table.table.abs_amnt = seq_table.sample_total.apply(target=seq_table.table.original)

    from .transform import ReactedFractionNormalizer
    reacted_frac = ReactedFractionNormalizer(input_samples=input_samples,
                                             reduce_method='median',
                                             remove_empty=True)
    seq_table.table.reacted_frac = reacted_frac.apply(seq_table.table.abs_amnt)
    from .filters import DetectedTimesFilter
    seq_table.table.seq_in_all_smpl_reacted_frac = DetectedTimesFilter(
        min_detected_times=seq_table.table.reacted_frac.shape[1]
    )(seq_table.table.reacted_frac)
    seq_table.truth = param_table

    if save_to is not None:
        from pathlib import Path
        save_path = Path(save_to)
        if save_path.suffix == '':
            save_path.mkdir(parents=True, exist_ok=True)
            total_amount.to_csv(f'{save_path}/dna_amount.csv')
            x.to_csv(f'{save_path}/x.csv')
            Y.to_csv(f'{save_path}/Y.csv')
            param_table.to_csv(f'{save_path}/truth.csv')
            seq_table.to_pickle(f"{save_path}/seq_table.pkl")
        else:
            logging.error('save_to should be a directory', error_type=TypeError)
    return x, Y, total_amount, param_table, seq_table


def get_pct_gaussian_error(rate):
    """Return a function to apply Gaussian error proportional to the value"""

    def pct_gaussian_error(amount):
        """For DNA amount, assign a given ratio Gaussian error"""
        return np.random.normal(loc=amount, scale=amount * rate)

    return pct_gaussian_error


@simu_doc.compose("""Deprecated. Simulate k-seq count dataset similar to the experimental condition of BYO-doped pool, that
    t: reaction time (90 min)
    alpha: degradation ratio of BYO (0.479)
    x_values: controlled BYO concentration points: 1 input pool with triple sequencing depth,
      5 BYO concentration with triplicates:
        [-1 (input pool),
         2e-6, 2e-6, 2e-6,
         10e-6, 10e-6, 10e-6,
         50e-6, 50e-6, 50e-6,
         250e-6, 250e-6, 250e-6,
         1260e-6, 1260e-6, 1260e-6]

Parameter for each sequences were sampled from given distribution defined from arguments
    - p0: log normal from exp(N(p0_loc, p0_scale))
    - k: log normal from k_95 95-percentile for k
    - A: uniform from [0, 1]

Other args:
<<uniq_seq_num, depth, total_amount_error, save_to>>
    plot_dist (bool): if pairwise figures of distribution for simulated parameters (p0, A, k, kA)

Returns:
<<x, Y, param_table, truth, seq_table>>
""")
def simulate_w_byo_doped_condition_from_param_dist(uniq_seq_num, depth, p0_loc, p0_scale, k_95,
                                                   total_dna_error_rate=0.1, seed=23,
                                                   save_to=None, plot_dist=True):

    c_list = [-1] + list(np.repeat(
        np.expand_dims([2e-6, 10e-6, 50e-6, 250e-6, 1250e-6], -1), 3
    ))

    N_list = [uniq_seq_num * depth if c >= 0 else uniq_seq_num * depth * 3 for c in c_list]

    if total_dna_error_rate is None:
        total_dna_error = 0
    else:
        total_dna_error = get_pct_gaussian_error(total_dna_error_rate)

    x, Y, dna_amount, truth, seq_table = simulate_counts(
        uniq_seq_num=uniq_seq_num,
        x_values=c_list,
        total_reads=N_list,
        p0_generator=DistGenerators.compo_lognormal(loc=p0_loc, scale=p0_scale, size=uniq_seq_num),
        k=DistGenerators.lognormal(c95=k_95, size=uniq_seq_num),
        A=DistGenerators.uniform(low=0, high=1, size=uniq_seq_num),
        total_amount_error=total_dna_error,
        reps=1,
        save_to=save_to,
        seed=seed
    )
    truth['kA'] = truth.k * truth.A

    if save_to:
        config = {
            'uniq_seq_num': uniq_seq_num,
            'depth': depth,
            'p0_loc': p0_loc,
            'p0_scale': p0_scale,
            'k_95': k_95,
            'total_dna_error_rate': total_dna_error_rate
        }
        from ..utility.file_tools import dump_json
        dump_json(config, save_to + '/config.txt')

    if plot_dist:
        from ..utility.plot_tools import pairplot
        pairplot(data=truth, vars_name=['p0', 'A', 'k', 'kA'],
                 vars_log=[True, False, True, True], diag_kind='kde')

    return x, Y, dna_amount, truth, seq_table


@simu_doc.compose("""Simulate k-seq count dataset based on the experimental condition of BYO-doped pool, that
    t: reaction time (90 min)
    alpha: degradation ratio of BYO (0.479)
    x_values: controlled BYO concentration points: 1 input pool with triple sequencing depth,
      5 BYO concentration with triplicates:
        [-1 (input pool),
         2e-6, 2e-6, 2e-6,
         10e-6, 10e-6, 10e-6,
         50e-6, 50e-6, 50e-6,
         250e-6, 250e-6, 250e-6,
         1260e-6, 1260e-6, 1260e-6]

Parameter for each sequences were sampled from previous point estimate results:
    - point_est_csv: load point estimates results to extract estimated k and A
    - seqtable_path: path to load input sample SeqData object to get p0 information

Returns:
<<x, Y, param_table, truth, seq_table>>
""")
def simulate_on_byo_doped_condition_from_exp_results(dataset, fitting_res, uniq_seq_num=None,
                                                     x_values=None, total_reads=None, sequencing_depth=40,
                                                     n_input=1, table_name='original',
                                                     total_dna_error_rate=0.1, seed=23,
                                                     plot_dist=False, save_to=None):
    from pathlib import Path, PosixPath

    if isinstance(dataset, str):
        from ..utility import file_tools
        dataset = file_tools.read_pickle(dataset)

    # parse fitting results
    if isinstance(fitting_res, (str, Path, PosixPath)):
        if Path(fitting_res).is_dir():
            result_table = pd.read_csv(Path(fitting_res).joinpath('fit_summary.csv'), index_col=0)
        elif Path(fitting_res).is_file():
            result_table = pd.read_csv(Path(fitting_res), index_col=0)
        else:
            logging.error('Unknown path to fitting results', ValueError)
    if 'ka' in result_table.columns:
        result_table = result_table.rename(columns={'ka': 'kA'})

    # add composition of initial pool (p0) to result table
    count_table = getattr(dataset.table, table_name)
    count_table = count_table.loc[result_table.index, dataset.grouper.input.group]
    p0 = (count_table / count_table.sum(axis=0)).mean(axis=1)
    result_table['p0'] = p0 / p0.sum()
    result_table = result_table[['k', 'A', 'p0', 'kA']]
    result_table = result_table[~result_table.isna().any(axis=1)]

    if x_values is None:
        x_values = getattr(dataset, 'x_values')
    x_values = pd.concat([pd.Series(np.repeat(-1, n_input), index=[f'input_{ix + 1}' for ix in range(n_input)]),
                         x_values])

    if uniq_seq_num is None:
        uniq_seq_num = result_table.shape[0]

    if total_reads is None:
        total_reads = [uniq_seq_num * sequencing_depth
                       if c >= 0 else uniq_seq_num * sequencing_depth * 3 for c in x_values]

    if total_dna_error_rate is None:
        total_amount_error = None
    else:
        total_amount_error = get_pct_gaussian_error(total_dna_error_rate)

    x, Y, dna_amount, truth, seq_table = simulate_counts(
        uniq_seq_num=uniq_seq_num,
        x_values=x_values,
        total_reads=total_reads,
        param_sample_from_df=result_table,
        total_amount_error=total_amount_error,
        weights=None,
        replace=uniq_seq_num >= result_table.shape[0],
        reps=1,
        save_to=save_to,
        seed=seed
    )
    truth['kA'] = truth.k * truth.A

    if save_to is not None:
        config = {
            'fitting_res': fitting_res,
            'dataset': str(dataset),
            'uniq_seq_num': uniq_seq_num,
            'depth': sequencing_depth,
            'total_dna_error_rate': total_dna_error_rate
        }
        from ..utility.file_tools import dump_json
        dump_json(config, save_to + '/config.txt')

    if plot_dist:
        from ..utility.plot_tools import pairplot
        pairplot(data=truth, vars_name=['p0', 'A', 'k', 'kA'],
                 vars_log=[True, False, True, True], diag_kind='kde')

    return x, Y, dna_amount, truth, seq_table


@simu_doc.compose("""Class to load simulation result""")
class SimulationResults:

    def __init__(self, dataset_dir, result_dir):
        """Survey estimation results
        - load fitting results from `result_dir/fit_summary.csv`
        - load truth and input count infor from `dataset_dir/truth.csv` and `input_counts`

        Optional to include:
        - input_counts: counts of sequences in the input pool
        - mean_counts: mean counts in all samples (input and reacted)

        Return:
            results: table of estimated k, A, kA
            truth: table of true k, A, p0, ka, and input_counts
            seq_list: list of indices of sequences that were able to estimate
        """

        allowed_col = [
            'k', 'A', 'kA',
            'A_mean', 'k_mean', 'kA_mean',
            'A_std', 'k_std', 'kA_std',
            'A_2.5%', 'k_2.5%', 'kA_2.5%',
            'A_50%', 'k_50%', 'kA_50%',
            'A_97.5%', 'k_97.5%', 'kA_97.5%',
            'bs_A_mean', 'bs_k_mean', 'bs_kA_mean',
            'bs_A_std', 'bs_k_std', 'bs_kA_std',
            'bs_A_2.5%', 'bs_k_2.5%', 'bs_kA_2.5%',
            'bs_A_50%', 'bs_k_50%', 'bs_kA_50%',
            'bs_A_97.5%', 'bs_k_97.5%', 'bs_kA_97.5%',
            'rep_A_mean', 'rep_k_mean', 'rep_kA_mean',
            'rep_A_std', 'rep_k_std', 'rep_kA_std'
        ]

        from pathlib import Path
        from ..utility.file_tools import read_pickle

        self.results = pd.read_csv(f'{result_dir}/fit_summary.csv', index_col=0)
        self.cols = self.results.columns[self.results.columns.isin(allowed_col)].values
        self.seq_list = self.results[~self.results[self.cols].isna().any(axis=1)].index.values
        self._bs_prefix = 'bs_' if 'bs_kA_2.5%' in self.results.columns else ''

        if Path(dataset_dir).is_file():
            dataset = read_pickle(dataset_dir)
        else:
            dataset = read_pickle(dataset_dir + '/seq_table.pkl')

        self.truth = dataset.truth
        self.truth['input_counts'] = dataset.table.original.reindex(self.truth.index).s0
        self.truth['mean_counts'] = dataset.table.original.reindex(self.truth.index).mean(axis=1)

        logging.info(f'{self.truth.shape[0]} sequences simulated, '
                     f'{self.results.shape[0]} fitted, '
                     f'{len(self.seq_list)} has valid results')

    def get_fold_range(self, param):
        """Return the ratio of 97.5-percentile to 2.5-percentile"""
        return self.results[self._bs_prefix + param + '97.5%'][self.seq_list] / \
               self.results[self._bs_prefix + param + '2.5%'][self.seq_list]

    def get_est_results(self, param, pred_type='point_est'):
        """Return the estimation (pred) and truth of given parameter"""
        if pred_type in ['pe', 'point_est', 'point est', 'point_estimation', 'point estimation']:
            pred = self.results[param]
        elif pred_type in ['mean', 'bs_mean', 'bootstrap_mean']:
            pred = self.results[self._bs_prefix + param + '_mean']
        elif pred_type in ['median', 'bs_median', 'bootstrap_median']:
            pred = self.results[self._bs_prefix + param + '_50%']
        elif pred_type in ['rep_mean', 'replicate_mean']:
            pred = self.results[self._bs_prefix + param + '_mean']
        else:
            logging.error("Unknown pred_type, choose from 'point_est', 'bs_mean', 'bs_median', 'rep_mean'", ValueError)

        truth = self.truth[param]
        return pd.DataFrame({'pred': pred[self.seq_list], 'truth': truth[self.seq_list]})

    def _get_bs_ci95_accuracy(self, param='kA'):
        """Return a dataframe include if the bootstrap result (95-percentile) includes the true value"""

        result = pd.concat([
            self.results.reindex(self.seq_list)[[self._bs_prefix + param + '_2.5%', self._bs_prefix + param + '_97.5%']],
            self.truth.reindex(self.seq_list)[[param, 'mean_counts', 'input_counts']].rename(
                columns={param: param + '_truth'})
        ], axis=1)

        result['fold_range'] = result[self._bs_prefix + param + '_97.5%'] / result[self._bs_prefix + param + '_2.5%']
        result['in_ci95'] = ((result[param + '_truth'] >= result[self._bs_prefix + param + '_2.5%']) &
                             (result[param + '_truth'] <= result[self._bs_prefix + param + '_97.5%'])).astype('int')
        return result

    def _get_bs_sd_accuracy(self, param='kA', scale=1.96):
        """Return a dataframe include if the standard deviation included from bootstrap includes the true value"""

        result = pd.concat([
            self.results.reindex(self.seq_list)[[self._bs_prefix + param + '_mean', 'bs_' + param + '_std']],
            self.truth.reindex(self.seq_list)[[param, 'mean_counts', 'input_counts']].rename(
                columns={param: param + '_truth'})
        ], axis=1)

        result['in_ci95'] = (
                (result[param + '_truth'] >= result[self._bs_prefix + param + '_mean'] -
                 scale * result[self._bs_prefix + param + '_std']) &
                (result[param + '_truth'] <= result[self._bs_prefix + param + '_mean'] +
                 scale * result[self._bs_prefix + param + '_std'])
        ).astype('int')
        return result

    def _get_rep_sd_accuracy(self, param='kA', scale=1.96):
        """Return a dataframe include if the standard deviation included from replicates includes the true value"""

        result = pd.concat([
            self.results.reindex(self.seq_list)[['rep_' + param + '_mean', 'rep_' + param + '_std']],
            self.truth.reindex(self.seq_list)[[param, 'mean_counts', 'input_counts']].rename(
                columns={param: param + '_truth'})
        ], axis=1)

        result['in_ci95'] = ((result[param + '_truth'] >= result['rep_' + param + '_mean'] -
                              scale * result['rep_' + param + '_std']) &
                             (result[param + '_truth'] <= result['rep_' + param + '_mean'] +
                              scale * result['rep_' + param + '_std'])).astype('int')
        return result

    def get_uncertainty_accuracy(self, param, pred_type='bs_ci95'):
        """Return the accuracy of uncertainty estimation if uncertainty range includes the truth"""
        if pred_type in ['bs_ci95', 'bootstrap_ci95']:
            return self._get_bs_ci95_accuracy(param=param)
        elif pred_type in ['bs_sd', 'bootstrap_sd']:
            return self._get_bs_sd_accuracy(param=param)
        elif pred_type in ['rep_sd']:
            return self._get_rep_sd_accuracy(param=param)
        else:
            logging.error("Unknown pred_type, choose from 'bs_ci95', 'bs_sd', 'rep_sd'")

