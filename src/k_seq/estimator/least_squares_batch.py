"""Least-squares fitting for a batch of sequences"""

from .least_squares import SingleFitter, FitResults


def _read_seq_json(json_path):
    """Read single fitting results from json file and return a summarized pd.Series"""
    fit_res = FitResults.from_json(json_path)
    return fit_res.to_series()


def _read_work_fn(seq):
    """Work function to read JSON results for each sequence"""

    res = _read_seq_json(seq[1])
    res.name = seq[0]
    return res


class BatchFitResults:
    """Parse, store, and visualize BatchFitter results
    Only save results (detached from each fitter), corresponding fitter should be found by sequence

    Attributes:
        fitter: proxy to the `BatchFitter`
        summary (`pd.DataFrame`): summarized results with each sequence as index
        bs_record (dict of pd.DataFrame): {seq: `SingleFitter.results.uncertainty.records`}
        conv_record (dict of pd.DataFrame): {seq: `SingleFitter.results.convergence.records}

    Methods:
        summary_to_csv: export summary dataframe as csv file
        to_json: preferred format to save results
        to_pickle: save results as pickled dictionary
        from_pickle: load bootstrapping results from picked dictionary
        from_folder: link results to a saved folder
        from_record: overall method to infer either load `BatchFitResults` from pickled or a folder
    """

    def __init__(self, fitter=None, result_path=None):
        """Init a BatchFitResults instance
        Args:
            fitter (`BatchFitter`): corresponding fitter
            result_path (str): optional, path to saved results
        """
        self.fitter = fitter
        self.bs_record = None
        self.conv_record = None
        self.summary = None
        self._result_path = result_path
        self._sep_files = None
        if result_path:
            self.parse_saved_results()

        # TODO: add visualization here

    @staticmethod
    def generate_summary(result_folder_path, n_core=1, save_to=None):
        """Generate a summary csv file from given result folder. This could be used if summary was not successfully
        generated during fitting

        Result folder should have a structure of:
          - seqs
            - [seq name or hash].json
            - [if hash] seq_to_hash.json

        Args:
            result_folder_path (str): path to the root of `results` folder
            n_core (int): number of threads to process in parallel. Default 1
            save_to (str): save CSV file to local path

        Returns:
            pd.DataFrame of summary
        """
        from pathlib import Path
        from ..utility.file_tools import get_file_list

        if Path(result_folder_path).joinpath('seqs').exists():
            seq_root = Path(result_folder_path).joinpath('seqs')
        else:
            seq_root = Path(result_folder_path)

        file_list = get_file_list(str(seq_root), full_path=False)
        if 'seq_to_hash.json' in file_list:
            from k_seq.utility.file_tools import read_json
            file_list = read_json(seq_root.joinpath('seq_to_hash.json'))
            file_list = ((key, seq_root.joinpath(f'{hash_}.json')) for key, hash_ in file_list.items())
        else:
            file_list = ((seq, seq_root.joinpath(f'{seq}.json')) for seq in file_list)

        if n_core > 1:
            import multiprocessing as mp
            pool = mp.Pool(n_core)
            result = pool.map(_read_work_fn, file_list)
        else:
            result = [_read_work_fn(seq) for seq in file_list]
        result = pd.DataFrame(result)
        if save_to is not None:
            result.to_csv(save_to)
        return result

    def parse_saved_results(self):
        """Load/link data from `self.result_path`
        TODO: Need to set internal trigger of how to load results
        """

        from pathlib import Path
        result_path = Path(self.result_path)
        if result_path.is_dir():
            if result_path.joinpath('results.pkl').exists():
                self.sep_files = False
                result_path = result_path.joinpath('results.pkl')
            elif result_path.joinpath('results').is_dir():
                self.sep_files = True
                result_path = result_path.joinpath('results/')
            elif result_path.joinpath('summary.pkl').exists():
                self.sep_files = True
                result_path = result_path
            else:
                raise ValueError('Can not parse result_path.')
        else:
            self.sep_files = False
            result_path = result_path
        from ..utility.file_tools import read_pickle
        if self.sep_files:
            # record the results root
            self.result_path = result_path
            self.summary = read_pickle(result_path.joinpath('summary.pkl'))
        else:
            results = read_pickle(result_path)
            self.summary = results['summary']
            if 'bs_record' in results.keys():
                self.bs_record = results['bs_record']

    @property
    def bs_record(self):
        if self.sep_files is True:
            print('Bootstrap records are saved as separate files, use `get_bs_record` instead')
        else:
            return self._bs_record

    @bs_record.setter
    def bs_record(self, value):
        self._bs_record = value
        self.sep_files = False

    def get_bs_record(self, seq=None):
        """Load bootstrap records for seq from files
        Args:
            seq (str or a list of str): a sequence or a list of sequence

        Returns:
            a pd.DataFrame of bootstraprecordsif seq is str
            a dict of pd.DataFrame contains bootstrap records if seq is a list of str
        """
        import pandas as pd
        import numpy as np
        from ..utility.file_tools import read_pickle

        if isinstance(seq, (list, pd.Series, np.ndarray)):
            return {seq: read_pickle(str(self.result_path) + '/' + s + '.pkl') for s in seq}
        else:
            return read_pickle(str(self.result_path) + '/' + seq + '.pkl')

    def summary_to_csv(self, path):
        """Save summary table as csv file"""
        self.summary.to_csv(path)

    def to_pickle(self, output_dir, bs_record=True, sep_files=True):
        """Save fitting results as a pickled dict, notice: `to_json` is preferred
        Args:
             output_dir (str): path to saved results, should be the parent of target location
             bs_record (bool): if output bs_record as well
             sep_files (bool): if save bs_records as separate files
                 If True:
                     |path/results/
                         |- summary.pkl
                         |- seqs
                             |- seq1.pkl
                             |- seq2.pkl
                              ...
                if False:
                     save to path/results.pkl contains
                     {
                         summary: pd.DataFrame
                         bs_records: {
                            seq1 (pd.DataFrame)
                            seq2 (pd.DataFrame)
                            ...
                       }
                     }
        """
        from ..utility.file_tools import dump_pickle

        if sep_files:
            check_dir(f'{output_dir}/results/')
            dump_pickle(obj=self.summary, path=f'{output_dir}/results/summary.pkl')
            if bs_record and self.bs_record is not None:
                [dump_pickle(obj=record, path=f'{output_dir}/results/{seq}.pkl')
                 for seq, record in self.bs_record.items()]
        else:
            check_dir(output_dir)
            data_to_dump = {'summary': self.summary}
            if bs_record and self.bs_record is not None:
                data_to_dump['bs_record'] = self.bs_record
            dump_pickle(obj=data_to_dump, path=output_dir + '/results.pkl')

    @classmethod
    def from_pickle(cls, path_to_pickle, fitter=None):
        """Create a `BatchFitResults` instance with results loaded from pickle
        Notice:
            this will take a very long time if the pickle is large
        """
        return cls(fitter=fitter, result_path=path_to_pickle)

    def to_json(self, output_dir, bs_record=True, sep_files=True):
        """Serialize results as json format
        Args:
             output_dir (str): path to save results, should be the parent of target location
             bs_record (bool): if output bs_record as well
             sep_files (bool): if save bs_records as separate files
                 If True:
                     |path/results/
                         |- summary.json
                         |- seqs
                             |- seq1.json
                             |- seq2.json
                              ...
                if False:
                     save to path/results.json contains
                     {
                         summary: pd.DataFrame.json
                         bs_records: {
                            seq1 (pd.DataFrame.json)
                            seq2 (pd.DataFrame.json)
                            ...
                       }
                     }
        """
        check_dir(output_dir)
        if sep_files:
            check_dir(f'{output_dir}/results/')
            dump_json(obj=self.summary.dump_json(), path=f'{output_dir}/results/summary.json')
            if bs_record and self.bs_record is not None:
                check_dir(f'{output_dir}/seqs')
                for seq, record in self.bs_record.items():
                    dump_json(obj=record.dump_json(), path=f"{output_dir}/results/seqs/{seq}.json")
        else:
            data_to_json = {'summary': self.summary.to_json()}
            if bs_record and self.bs_record is not None:
                data_to_json['bs_record'] = {seq: record.to_json() for seq, record in self.bs_record}
            dump_json(obj=data_to_json, path=f"{output_dir}/results.json")

    @classmethod
    def from_json(cls, fitter, json_o_path):
        """Load results from JSON
        TODO: parse JSON results
        """

        import pandas as pd
        import json
        try:
            # first consider it is a json string
            json_data = json.loads(json_o_path)
        except json.JSONDecodeError:
            try:
                with open(json_o_path, 'r') as handle:
                    json_data = json.load(handle)
            except:
                raise TypeError("Invalid JSON input")
        inst = cls(fitter=fitter)
        inst.summary = pd.read_json(json_data['summary'])
        inst.bs_record = {key: pd.read_json(record) for key, record in json_data['bs_record']}

        return inst


def _work_fn(worker, point_estimate, bootstrap, convergence_test):
    """Utility work function to parallelize workers"""
    worker.fit(point_estimate=point_estimate, bootstrap=bootstrap, convergence_test=convergence_test)
    return worker


class BatchFitter(EstimatorType):
    """Fitter for least squared batch fitting

    Attributes:
        y_data_batch (pd.DataFrame): a table containing y values to fit (e.g. reacted fraction in each sample)
    {attr}
        note (str): note about this fitting job
        fitters (None or dict): a dictionary of SingleFitters if `self.config.keep_single_fitters` is True
        seq_to_fit (list of str): list of seq to fit for this job
        results (BatchFitResult): fitting results
        fit_params (AttrScope): parameters pass to each fitting, should be same for each sequence, includes:
    {fit_params}
    """.format(attr=doc_helper.get(['model', 'x_data', 'parameters'], indent=4),
               fit_params=doc_helper.get(['x_data', 'model', 'parameters', 'bounds', 'init_guess', 'opt_method',
                                          'exclude_zero', 'metrics', 'rnd_seed', 'bootstrap_num', 'bs_record_num',
                                          'bs_method', 'curve_fit_kwargs', 'silent']))

    def __repr__(self):
        from ..utility.func_tools import get_object_hex
        return f'Least-squared BatchFitter at {get_object_hex(self)}'

    def __init__(self, y_data_batch, x_data, model, sigma=None, bounds=None, seq_to_fit=None,
                 bootstrap_num=0, bs_record_num=0, bs_method='pct_res', grouper=None,
                 conv_reps=0, init_range=None,
                 opt_method='trf', exclude_zero=False, init_guess=None, metrics=None, rnd_seed=None,
                 curve_fit_kwargs=None, note=None, result_path=None):
        """
        Args:
            y_data_batch (pd.DataFrame or str): a set of y_data to fit form rows of y_data_batch, can be a string
                indicate the path to a pickled pd.DataFrame record
        {args}
            note (str): Optional notes for the fitter
            results: a proxy to BatchFitResults
        """.format(args=doc_helper.get(['x_data', 'model', 'bounds', 'sigma', 'bootstrap_num', 'bs_record_num',
                                        'bs_method', 'grouper', 'opt_method', 'exclude_zero', 'init_guess',
                                        'metrics', 'rnd_seed', 'curve_fit_kwargs', 'seq_to_fit'], indent=4))

        from ..utility.func_tools import get_func_params, AttrScope
        super().__init__()

        logging.info('Creating the BatchFitter...')

        self.model = model
        self.parameters = get_func_params(model, exclude_x=True)
        self.note = note

        # parse y_data_batch
        from ..utility.file_tools import table_object_to_dataframe
        self.y_data_batch = table_object_to_dataframe(y_data_batch)

        # process seq_to_fit
        if seq_to_fit is not None:
            if isinstance(seq_to_fit, (list, np.ndarray, pd.Series)):
                self.seq_list = list(seq_to_fit)
            else:
                raise TypeError('Unknown seq_to_fit type, is it list-like?')
        self.seq_to_fit = seq_to_fit

        # prep fitting params shared by all fittings
        if isinstance(x_data, pd.Series):
            self.x_data = x_data[y_data_batch.columns.values]
        elif len(x_data) != y_data_batch.shape[1]:
            raise ValueError('x_data length and table column number does not match')
        else:
            self.x_data = np.array(x_data)

        if sigma is not None:
            if np.shape(sigma) != np.shape(self.y_data_batch):
                raise ValueError('Shape of sigma does not match the shape of y_data_batch')
        self.sigma = sigma

        if bounds is None:
            bounds = (-np.inf, np.inf)

        if len(x_data) <= 1:
            logging.warning("Number of data points less than 2, bootstrap will not be performed")
            bootstrap_num = 0
        self.bootstrap = bootstrap_num > 0

        # contains parameters should pass to the single fitter
        self.fit_params = AttrScope(
            x_data=self.x_data,
            model=self.model,
            parameters=self.parameters,
            bounds=bounds,
            opt_method=opt_method,
            bootstrap_num=bootstrap_num,
            bs_record_num=bs_record_num,
            bs_method=bs_method,
            conv_reps=conv_reps,
            init_range=init_range,
            exclude_zero=exclude_zero,
            init_guess=init_guess,
            metrics=metrics,
            rnd_seed=rnd_seed,
            grouper=grouper if bs_method == 'stratified' else None,
            curve_fit_kwargs=curve_fit_kwargs,
            silent=True
        )

        self.results = BatchFitResults(fitter=self, result_path=result_path)

        # TODO: recover the visualizer
        # from .visualizer import fitting_curve_plot, bootstrap_params_dist_plot, param_value_plot
        # from ..utility import FunctionWrapper
        # self.visualizer = FunctionWrapper(data=self,
        #                                   functions=[
        #                                       fitting_curve_plot,
        #                                       bootstrap_params_dist_plot,
        #                                       param_value_plot
        #                                   ])
        logging.info('BatchFitter created')

    # def generate_summary(self, result_folder_path):
    #     """Generate a summary csv file from given result folder. This could be used if summary was not successfully
    #     generated during fitting

        # Result folder should have a structure of:
        #   - seqs
        #     - [seq name or hash].json
        #     - [if hash] seq_to_hash.json

        # Args:
        #     result_folder_path (str): path to the root of `results` folder

        # Returns:
        #     pd.DataFrame of summary
        # """
        # from pathlib import Path
        # from ..utility.file_tools import get_file_list
        # if Path(result_folder_path).append('seqs').exists():




    def worker_generator(self, stream_to_disk=None, overwrite=False):
        """Return a generator of worker for each sequence"""

        if self.seq_to_fit is None:
            seq_list = self.y_data_batch.index.values
        else:
            seq_list = self.seq_to_fit
        for seq in seq_list:
            try:
                yield SingleFitter(
                    name=seq,
                    y_data=self.y_data_batch.loc[seq],
                    sigma=None if self.sigma is None else self.sigma.loc[seq],
                    save_to=None if stream_to_disk is None else f"{stream_to_disk}/seqs/{seq}.json",
                    overwrite=overwrite,
                    **self.fit_params.__dict__
                )
            except:
                raise Exception(f'Can not create fitting worker for {seq}')

    def fit(self, deduplicate=False, parallel_cores=1,
            point_estimate=True, bootstrap=False, convergence_test=False,
            stream_to_disk=None, overwrite=False):
        """Run the estimation
        Args:
            deduplicate (bool): hash the y_data_batch to deduplicate before fitting if True
            parallel_cores (int): number of parallel cores to use. Default 1
            point_estimate (bool): if do point estimation, default True
            bootstrap (bool): if do bootstrap uncertainty estimation, default False
            convergence_test (bool): if do convergence test, default False
            stream_to_disk (str): Directly stream fitting results to disk if output path is given
                will create a folder with name of seq/hash with pickled dict of fitting results
            overwrite (bool): if overwrite existing results when stream to disk. Default False.
        """

        logging.info('Batch fitting starting...')

        if deduplicate:
            self._hash()
            if stream_to_disk:
                check_dir(stream_to_disk + '/seqs/')
                dump_json(obj=self._seq_to_hash, path=f"{stream_to_disk}/seqs/seq_to_hash.json")

        from functools import partial
        work_fn = partial(_work_fn, point_estimate=point_estimate,
                          bootstrap=bootstrap, convergence_test=convergence_test)
        worker_generator = self.worker_generator(stream_to_disk=stream_to_disk, overwrite=overwrite)
        if parallel_cores > 1:
            import multiprocessing as mp
            pool = mp.Pool(processes=int(parallel_cores))
            logging.info('Use multiprocessing to fit in {} parallel threads...'.format(parallel_cores))
            workers = pool.map(work_fn, worker_generator)
        else:
            # single thread
            logging.info('Fitting in a single thread...')
            workers = [work_fn(fitter) for fitter in worker_generator]

        # record results
        if self.bootstrap:
            self.results.bs_record = {worker.name: worker.results.uncertainty.records for worker in workers}
        if convergence_test:
            self.results.conv_record = {worker.name: worker.results.convergence.records for worker in workers}
        self.results.summary = pd.DataFrame({worker.name: worker.summary() for worker in workers}).transpose()

        if deduplicate:
            self._hash_inv()
        logging.info('Fitting finished')

    def summary(self, save_to=None):
        if save_to is None:
            return self.results.summary
        else:
            self.results.summary.to_csv(save_to)

    def _hash(self):
        """De-duplicate rows before fitting"""

        def hash_series(row):
            return hash(tuple(row))

        self._y_data_batch_dup = self.y_data_batch.copy()
        if self.seq_to_fit is not None:
            # filter the seq to fit
            self._seq_to_fit_dup = self.seq_to_fit.copy()
            self.y_data_batch = self.y_data_batch.loc[self.seq_to_fit]
        # find seq to hash mapping
        self._seq_to_hash = self.y_data_batch.apply(hash_series, axis=1).to_dict()
        # only keep the first instance of each hash
        self.y_data_batch = self.y_data_batch[~self.y_data_batch.duplicated(keep='first')]
        if isinstance(self.sigma, pd.DataFrame):
            # only accept sigma as an pd.DataFrame
            self._sigma_dup = self.sigma.copy()
            # filter sigma table for only the first instance of each hash
            self.sigma = self.sigma.loc[self.y_data_batch.index]
            # convert seq --> hash
            self.sigma.rename(index=self._seq_to_hash, inplace=True)
        # convert seq --> hash
        self.y_data_batch.rename(index=self._seq_to_hash, inplace=True)
        if self.seq_to_fit is not None:
            self.seq_to_fit = [self._seq_to_hash[seq] for seq in self.seq_to_fit]
        logging.info('Shrink rows in table by removing duplicates: '
                     f'{self._y_data_batch_dup.shape[0]} --> {self.y_data_batch.shape[0]}')

    def _hash_inv(self):
        """Recover the hashed results"""

        logging.info('Recovering original table from hash...')

        def get_summary(seq):
            return self.results.summary.loc[self._seq_to_hash[seq]]

        # map hash --> seq for results summary
        self.results.summary = pd.Series(data=list(self._seq_to_hash.keys()),
                                         index=list(self._seq_to_hash.keys())).apply(get_summary)
        # map hash --> seq for bs_record
        if self.results.bs_record is not None:
            self.results.bs_record = {seq: self.results.bs_record[seq_hash]
                                      for seq, seq_hash in self._seq_to_hash.items()}
        # recover the original y_data_batch
        self.y_data_batch = self._y_data_batch_dup.copy()
        del self._y_data_batch_dup
        # recover the original sigma if exists
        if hasattr(self, '_sigma_dup'):
            self.sigma = self._sigma_dup.copy()
            del self._sigma_dup
        # recover the original seq_to_fit if exists
        if hasattr(self, '_seq_to_fit'):
            self.seq_to_fit = self._seq_to_fit_dup.copy()
            del self._seq_to_fit_dup

    def save_model(self, output_dir, results=True, bs_results=True, sep_files=True, tables=True):
        """Save model to a given directory
        model_config will be saved as a pickled dictionary to recover the model
            - except for `y_data_batch` and `sigma` which are too large

        Args:
            output_dir (str): path to save the model, create if the path does not exist
            results (bool): if save estimation results to `results` as well, to be load by `BatchFitResults`,
                Default True
            bs_results (bool): if save bootstrap results
            sep_files (bool): if save the record of bootstrap as separate files in a subfolder `results/seqs/`
                Default True
            tables (bool): if save tables (y_data_batch, sigma) in the folder. Default True
        """
        from ..utility.file_tools import dump_pickle

        check_dir(output_dir)
        dump_pickle(
            obj={
                **{'parameters': self.parameters,
                   'note': self.note,
                   'seq_to_fit': self.seq_to_fit},
                **self.fit_params.__dict__
            },
            path=str(output_dir) + '/model_config.pkl'
        )
        if results:
            self.save_results(result_path=str(output_dir), bs_results=bs_results, sep_files=sep_files)
        if tables is not None:
            dump_pickle(obj=self.y_data_batch, path=str(output_dir) + '/y_data.pkl')
            if self.sigma is not None:
                dump_pickle(obj=self.sigma, path=str(output_dir) + '/sigma.pkl')

    def save_results(self, result_path, bs_results=True, sep_files=True, use_pickle=False):
        """Save results to disk as JSON or pickle
        JSON is preferred for speed, readability, compatibility, and security
        """
        if use_pickle:
            self.results.to_pickle(result_path, bs_record=bs_results, sep_files=sep_files)
        else:
            self.results.to_json(result_path, bs_record=bs_results, sep_files=sep_files)

    @classmethod
    def load_model(cls, model_path, y_data_batch=None, sigma=None, result_path=None):
        """Create a model from pickled config file

        Args:
            model_path (str): path to picked model configuration file or the saved folder
            y_data_batch (pd.DataFrame or str): y_data table for fitting
            sigma (pd.DataFrame or str): optional sigma table for fitting
            result_path (str): path to fitting results

        Returns:
            a BatchFitter instance
        """

        from ..utility.file_tools import read_pickle
        from pathlib import Path

        config_file = model_path if Path(model_path).is_file() else model_path + '/model_config.pkl'
        model_config = read_pickle(config_file)
        if y_data_batch is None:
            # try infer from the folder
            y_data_batch = read_pickle(model_path + '/y_data.pkl')
        else:
            if isinstance(y_data_batch, str):
                y_data_batch = read_pickle(y_data_batch)
        if sigma is not None:
            if isinstance(sigma, str):
                sigma = read_pickle(sigma)
        return cls(y_data_batch=y_data_batch, sigma=sigma, result_path=result_path, **model_config)


def load_estimation_results(point_est_csv=None, seqtable=None, bootstrap_csv=None,
                            **kwargs):
    """Collect estimation results from multiple resources (e.g. summary.csv files) and compose a summary table
    Sequences will be the union of indices in point estimate, bootstrap, and convergence test if avaiable

    Resources:
      - count_table/seq_table: input counts, mean counts
      - point estimates: point estimation for parameters and metrics
      - bootstrap: uncertainty estimation from bootstrap
      - convergence test: convergence tests results

    Args:
        seq_table (str): path to pickled `SeqData` or `pd.DataFrame` object,
            will import 'input_counts'/, 'mean_counts'
        point_est_csv (str): optional, path to reported csv file from point estimation
        seqtable_path (str): optional. path to original seqTable object for count info
        bootstrap_csv (str): optional. path to csv file from bootstrap
        kwargs: optional keyword argument of callable to calculate extra columns, apply on results dataframe row-wise

    Returns:
        a pd.DataFrame contains composed results from provided information

    """

    point_est_res = pd.read_csv(point_est_csv, index_col=0)
    est_res = point_est_res[point_est_res.columns]
    seq_list = est_res.index.values

    if seqtable_path:
        # add counts in input pool
        from ..utility import file_tools
        seq_table = file_tools.read_pickle(seqtable_path)
        if seq_table.grouper and hasattr(seq_table.grouper, 'input'):
            est_res['input_counts'] = seq_table.table[seq_table.grouper.input.group].loc[seq_list].mean(axis=1)
        est_res['mean_counts'] = seq_table.table.loc[seq_list].mean(axis=1)
        est_res['min_counts'] = seq_table.table.loc[seq_list].min(axis=1)

        if hasattr(seq_table, 'pool_peaks'):
            # has doped pool, add dist to center
            from ..data import landscape
            mega_peak = landscape.Peak.from_peak_list(seq_table.pool_peaks)
            est_res['dist_to_center'] = mega_peak.dist_to_center

    if bootstrap_csv:
        bootstrap_res = pd.read_csv(bootstrap_csv, index_col=0)
        # add bootstrap results
        est_res[['kA_mean', 'kA_std', 'kA_2.5%', 'kA_50%', 'kA_97.5%']] = bootstrap_res[
            ['kA_mean', 'kA_std', 'kA_2.5%', 'kA_50%', 'kA_97.5%']]
        est_res['A_range'] = bootstrap_res['A_97.5%'] - bootstrap_res['A_2.5%']

    if convergence_csv:
        pass

    if kwargs:
        for key, func in kwargs.items():
            if callable(func):
                est_res[key] = est_res.apply(func, axis=1)
            else:
                logging.error(f'Keyword argument {key} is not a function', error_type=TypeError)
    return est_res
