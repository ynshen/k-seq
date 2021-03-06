"""Least-squares fitting for a batch of sequences"""

from yutility import logging
from .least_squares import SingleFitter, FitResults, doc_helper
from ._estimator import Estimator
import numpy as np
import pandas as pd
import ctypes
from ..utility.file_tools import check_dir, read_json, dump_json, dump_pickle, read_pickle
from pathlib import Path

info = logging.info

__all__ = ['BatchFitter', 'BatchFitResults']


def _work_fn(worker, point_estimate, bootstrap, convergence_test, replicates):
    """Utility work function to parallelize workers"""
    worker.fit(point_estimate=point_estimate, bootstrap=bootstrap,
               convergence_test=convergence_test, replicates=replicates)
    logging.debug(f'\nFit sequence: {worker.name}')
    logging.debug(worker.x_data)
    logging.debug(worker.y_data)
    return worker


doc_helper.add(
    y_dataframe=('pd.DataFrame', 'Table of y values for sequences (rows) to fit kinetic models'),
    seq_to_fit=('int or list of seq', 'pick top n sequences in the table for fitting or only fit selected sequences')
)


@doc_helper.compose("""Least-squares fitting for batch of sequences

Attributes:
    <<y_dataframe, model, x_data, seq_to_fit, sigma>>
    note (str): note about this fitting job
    results (BatchFitResult): accessor to fitting results
    fit_params (AttrScope): collection of arguments pass to each single seq fitting, includes:
        <<x_data, model, bounds, init_guess, opt_method, exclude_zero, metrics, rnd_seed, curve_fit_kwargs, 8>>
        <<replicates, bootstrap_num, bs_record_num, bs_method, bs_stats, grouper, record_full, 8>>
        <<conv_reps, conv_init_range, conv_stats, 8>>
        <<overwrite, 8>>
""")
class BatchFitter(Estimator):

    def __repr__(self):
        return f'Least-squared BatchFitter at {super().__repr__()}'

    @doc_helper.compose("""Initialize a BatchFitter
    
    Args:
        y_dataframe (pd.DataFrame or str)
        <<y_dataframe, x_data, model, seq_to_fit, sigma, bounds, init_guess, opt_method, exclude_zero>>
        <<metrics, rnd_seed, curve_fit_kwargs>>
        <<bootstrap_num, bs_record_num, bs_method, bs_stats, grouper, record_full>>
        <<conv_reps, conv_init_range, conv_stats>>
        note (str): optional notes for the estimator
        large_dataset (bool: if trigger strategy to work on large dataset (e.g. > 1000 seqs). If True, deduplicate of 
            sequences with same reacted fractions in each concentration will be performed and results will be streamed 
            to hard drive 
        result_path (str): if not None, load results from the path
    """)
    def __init__(self, y_dataframe, x_data, model,
                 x_label=None, y_label=None,
                 seq_to_fit=None, sigma=None, bounds=None, init_guess=None,
                 opt_method='trf', exclude_zero=False, metrics=None, rnd_seed=None, curve_fit_kwargs=None,
                 replicates=None,
                 bootstrap_num=0, bs_record_num=0, bs_method='pct_res', bs_stats=None, grouper=None, record_full=False,
                 conv_reps=0, conv_init_range=None, conv_stats=None,
                 note=None, large_dataset=False, verbose=1, result_path=None):

        from ..utility.func_tools import AttrScope, get_func_params

        super().__init__()

        logging.info('Creating the BatchFitter...')

        self.model = model
        self.note = note

        # parse y_dataframe
        from ..utility.file_tools import table_object_to_dataframe
        self.y_dataframe = table_object_to_dataframe(y_dataframe)

        # process seq_to_fit
        if seq_to_fit is not None:
            if isinstance(seq_to_fit, (list, np.ndarray, pd.Series)):
                self.seq_to_fit = list(seq_to_fit)
            elif isinstance(seq_to_fit, int):
                self.seq_to_fit = y_dataframe.index[:seq_to_fit].values
            else:
                logging.error('Unknown seq_to_fit type, is it list-like or int?', error_type=TypeError)
        else:
            self.seq_to_fit = seq_to_fit

        # prep fitting params shared by all fittings
        if isinstance(x_data, pd.Series):
            self.x_data = x_data[y_dataframe.columns.values]
        elif len(x_data) != y_dataframe.shape[1]:
            logging.error('x_data length and table column number does not match', error_type=ValueError)
        else:
            self.x_data = np.array(x_data)

        if sigma is not None:
            if np.shape(sigma) != np.shape(self.y_dataframe):
                logging.error('Shape of sigma does not match the shape of y_dataframe', error_type=ValueError)
        self.sigma = sigma

        if bounds is None:
            bounds = (-np.inf, np.inf)

        if len(x_data) <= 1:
            logging.warning("Number of data points less than 2, bootstrap will not be performed")
            bootstrap_num = 0
        self.bootstrap = bootstrap_num > 0

        # contains arguments should pass to the single estimator
        self.fit_params = AttrScope(
            x_data=self.x_data,
            x_label=x_label,
            y_label=y_label,
            model=self.model,
            bounds=bounds,
            init_guess=init_guess,
            opt_method=opt_method,
            exclude_zero=exclude_zero,
            metrics=metrics,
            rnd_seed=rnd_seed,
            curve_fit_kwargs=curve_fit_kwargs,
            replicates=replicates,
            bootstrap_num=bootstrap_num,
            bs_record_num=bs_record_num,
            bs_method=bs_method,
            bs_stats=bs_stats,
            grouper=grouper if bs_method == 'stratified' else None,
            record_full=record_full,
            conv_reps=conv_reps,
            conv_init_range=conv_init_range,
            conv_stats=conv_stats,
            verbose=verbose,
        )
        if result_path is None:
            self.results = BatchFitResults(estimator=self)
        else:
            self.results = BatchFitResults.load_result(result_path)
        self.large_dataset = large_dataset
        self.results.large_dataset = large_dataset
        self.workers = None

        logging.info('BatchFitter created')

    def _worker_generator(self, stream_to=None, overwrite=False):
        """Return a generator of worker for each sequence"""

        if self.seq_to_fit is None:
            seq_list = self.y_dataframe.index.values
        else:
            seq_list = self.seq_to_fit
        for seq in seq_list:
            yield SingleFitter(
                name=seq,
                y_data=self.y_dataframe.loc[seq],
                sigma=None if self.sigma is None else self.sigma.loc[seq],
                save_to=None if stream_to is None else f"{str(stream_to)}/seqs/{seq}.json",
                overwrite=overwrite,
                **self.fit_params.__dict__
            )

    def fit(self, parallel_cores=1, point_estimate=True, replicates=False, bootstrap=False, convergence_test=False,
            stream_to=None, overwrite=False):
        """Run the estimation
        Args:
            parallel_cores (int): number of parallel cores to use. Default 1
            point_estimate (bool): if perform point estimation, default True
            bootstrap (bool): if perform bootstrap uncertainty estimation, default False
            replicates (bool): if perform replicates for uncertainty estimation, default False
            convergence_test (bool): if perform convergence test, default False
            stream_to (str): Directly stream fitting results to disk if output path is given
                will create a folder with name of seq/hash with pickled dict of fitting results
            overwrite (bool): if overwrite existing results when stream to disk. Default False.
        """

        from yutility.log import Timer
        logging.info('Batch fitting starting...')

        with Timer():
            if self.large_dataset and stream_to is None:
                logging.error('You are working with large dataset and stream_to needs to be specified',
                              error_type=ValueError)
            if not self.large_dataset and stream_to is not None:
                self.large_dataset = True
                logging.warning("You provided `stream_to` so the large_dataset method is used")

            if self.large_dataset:
                self._hash()
                self.results.result_path = Path(stream_to)
                check_dir(self.results.result_path.joinpath('seqs'))
                dump_json(obj=self._seq_to_hash, path=self.results.result_path.joinpath('seqs', 'seq_to_hash.json'))

            from functools import partial
            work_fn = partial(_work_fn, point_estimate=point_estimate, replicates=replicates,
                              bootstrap=bootstrap, convergence_test=convergence_test)
            worker_generator = self._worker_generator(stream_to=stream_to, overwrite=overwrite)
            if parallel_cores > 1:
                import multiprocessing as mp
                pool = mp.Pool(processes=int(parallel_cores))
                logging.info('Use multiprocessing to fit in {} parallel threads...'.format(parallel_cores))
                workers = pool.map(work_fn, worker_generator)
            else:
                # single thread
                logging.info('Fitting in a single thread...')
                workers = [work_fn(worker) for worker in worker_generator]

            # print(workers[0].summary())
            self.results.summary = pd.DataFrame({worker.name: worker.summary() for worker in workers}).transpose()
            self.results.summary.index.name = 'seq'
            # record result
            if self.bootstrap:
                if self.large_dataset:
                    self.results._bs_record = self._seq_to_hash
                else:
                    self.results._bs_record = {worker.name: worker.results.uncertainty.records for worker in workers}
            if convergence_test:
                if self.large_dataset:
                    self.results._conv_record = self._seq_to_hash
                else:
                    self.results._conv_record = {worker.name: worker.results.convergence.records for worker in workers}

            if self.large_dataset:
                self._hash_inv()
                self.results.to_json(output_dir=stream_to)

            logging.info('Fitting finished')

    def summary(self, save_to=None):
        if save_to is None:
            return self.results.summary
        else:
            self.results.summary.index.name = 'seq'
            self.results.summary.to_csv(save_to)

    def _hash(self):
        """De-duplicate rows before fitting"""

        def hash_series(row):
            """Return a positive hash value"""

            return ctypes.c_size_t(hash(tuple(row)))

        self._y_dataframe_dup = self.y_dataframe.copy()
        if self.seq_to_fit is not None:
            # filter the seq to fit
            self._seq_to_fit_dup = self.seq_to_fit.copy()
            self.y_dataframe = self.y_dataframe.loc[self.seq_to_fit]
        # find seq to hash mapping
        self._seq_to_hash = self.y_dataframe.apply(hash_series, axis=1).to_dict()
        # only keep the first instance of each hash
        self.y_dataframe = self.y_dataframe[~self.y_dataframe.duplicated(keep='first')]
        if isinstance(self.sigma, pd.DataFrame):
            # only accept sigma as an pd.DataFrame
            self._sigma_dup = self.sigma.copy()
            # filter sigma table for only the first instance of each hash
            self.sigma = self.sigma.loc[self.y_dataframe.index]
            # convert seq --> hash
            self.sigma.rename(index=self._seq_to_hash, inplace=True)
        # convert seq --> hash
        self.y_dataframe.rename(index=self._seq_to_hash, inplace=True)
        if self.seq_to_fit is not None:
            self.seq_to_fit = [self._seq_to_hash[seq] for seq in self.seq_to_fit]
        logging.info('Shrink rows in table by removing duplicates: '
                     f'{self._y_dataframe_dup.shape[0]} --> {self.y_dataframe.shape[0]}')

    def _hash_inv(self):
        """Recover the hashed results"""

        logging.info('Recovering original table from hash...')

        def get_summary(seq):
            return self.results.summary.loc[self._seq_to_hash[seq]]

        # map hash --> seq for results summary
        self.results.summary = pd.Series(data=list(self._seq_to_hash.keys()),
                                         index=list(self._seq_to_hash.keys())).apply(get_summary)

        # recover the original y_dataframe
        self.y_dataframe = self._y_dataframe_dup.copy()
        del self._y_dataframe_dup
        # recover the original sigma if exists
        if hasattr(self, '_sigma_dup'):
            self.sigma = self._sigma_dup.copy()
            del self._sigma_dup
        # recover the original seq_to_fit if exists
        if hasattr(self, '_seq_to_fit'):
            self.seq_to_fit = self._seq_to_fit_dup.copy()
            del self._seq_to_fit_dup

    def save_model(self, output_dir, results=True, bs_record=True, conv_record=True, tables=True):
        """Save model to a given directory
        model_config will be saved as a pickled dictionary to recover the model
            - except for `y_dataframe` and `sigma` which are too large

        Args:
            output_dir (str): path to save the model, create if the path does not exist
            results (bool): if save estimation results to `results` as well, to be load by `BatchFitResults`,
                Default True
            bs_record (bool): if save bootstrap records, default True
            conv_record (bool): if save convergence records, default True
            tables (bool): if save tables (y_dataframe, sigma) in the folder. Default True
        """

        check_dir(output_dir)
        dump_pickle(
            obj={
                **{'note': self.note,
                   'seq_to_fit': self.seq_to_fit},
                **self.fit_params.__dict__
            },
            path=str(output_dir) + '/model_config.pkl'
        )
        if results:
            self.save_results(result_path=str(output_dir), bs_record=bs_record, conv_record=conv_record)
        if tables is not None:
            dump_pickle(obj=self.y_dataframe, path=str(output_dir) + '/y_data.pkl')
            if self.sigma is not None:
                dump_pickle(obj=self.sigma, path=str(output_dir) + '/sigma.pkl')

    def save_results(self, result_path, bs_record=True, conv_record=True):
        """Save results to disk as JSON or pickle
        JSON is preferred for speed, readability, compatibility, and security
        """
        if self.large_dataset:
            self.results.to_json(result_path)
        else:
            self.results.to_pickle(Path(result_path).joinpath('results.pkl'),
                                   bs_record=bs_record, conv_record=conv_record)

    @classmethod
    def load_model(cls, model_path, y_dataframe=None, sigma=None, result_path=None):
        """Create a model from pickled config file

        Args:
            model_path (str): path to picked model configuration file or the saved folder
            y_dataframe (pd.DataFrame or str): y_data table for fitting
            sigma (pd.DataFrame or str): optional sigma table for fitting
            result_path (str): path to fitting results

        Returns:
            a BatchFitter instance
        """

        config_file = model_path if Path(model_path).is_file() else model_path + '/model_config.pkl'
        model_config = read_pickle(config_file)
        if y_dataframe is None:
            # try infer from the folder
            y_dataframe = read_pickle(model_path + '/y_data.pkl')
        else:
            if isinstance(y_dataframe, str):
                y_dataframe = read_pickle(y_dataframe)
        if sigma is not None:
            if isinstance(sigma, str):
                sigma = read_pickle(sigma)
        return cls(y_dataframe=y_dataframe, sigma=sigma, result_path=result_path, **model_config)


class BatchFitResults:
    """Parse, store, and visualize BatchFitter results
    Only save results (separate from each estimator), corresponding estimator should be found by sequence
    We used two data storage strategies:
        1. smaller dataset that was saved as ``results.pkl``: the pickled file is passed, and the results will be
            loaded to self.summary, self.bs_record, self.conv_record
        2. larger dataset that was saved in ``results/`` folder: self. summary will be loaded, self.bs_record and
            self.conv_record will be linked

    Attributes:
        estimator: proxy to the `BatchFitter`
        model (callable): model function
        data (AttrScope): contains data information:
            x_data (list-like): x_data for fitting
            y_dataframe (pd.DataFrame): a table of y_data for sequences
            simga (pd.DataFrame): a table representing each sigma for sequence in fitting
        large_data (bool): if True, it will not load all bootstrap or convergence results
        summary (`pd.DataFrame`): summarized results with each sequence as index

    Methods:
        bs_record: get bootstrap results {seq: `SingleFitter.results.uncertainty.records`}
        conv_record: {seq: `SingleFitter.results.convergence.records}
        summary_to_csv: export summary dataframe as csv file
        to_json: storage strategy for large files: save results as a folder of json files
        to_pickle: storage strategy for small files: save results as pickled dictionary
        from_pickle: load from a picked dictionary
        from_json: load from a folder of json files
        load_result: overall method to infer either load `BatchFitResults` from pickled or a folder

    Analysis:

    """

    def __init__(self, estimator=None, model=None, x_data=None, y_dataframe=None, sigma=None):
        """Init a BatchFitResults instance
        Args:
            estimator (`BatchFitter`): corresponding estimator
        """
        from ..utility import func_tools
        self.estimator = estimator
        self.model = model
        self.data = func_tools.AttrScope(x_data=x_data, y_dataframe=y_dataframe, sigma=sigma)
        self._bs_record = None
        self._conv_record = None
        self.summary = None
        self.result_path = None
        self.large_dataset = False

    def get_FitResult(self, seq=None):
        """Get FitResults from a JSON file
        """
        
        from .least_squares import FitResults
        if self._bs_record is None:
            logging.error('No bootstrap or convergence test record found', error_type=TypeError)
        else:
            seq_to_hash = self._bs_record

        if seq is None:
            return seq_to_hash

        if isinstance(seq_to_hash[seq], (list, tuple)):
            # new hierarchical format
            tg_ix, hash_ix = seq_to_hash[seq]
            result = FitResults.from_json(json_path=f'{hash_ix}.json',
                                          tarfile=self.result_path.joinpath('seqs', f'{tg_ix}.tar.gz'))
        else:
            # old format
            if self.result_path.joinpath('seqs', f"{seq_to_hash[seq]}.json").exists():
                logging.info(f"load result from {seq_to_hash[seq]}.json")
                result = FitResults.from_json(self.result_path.joinpath('seqs', f'{seq_to_hash[seq]}.json'))
            elif self.result_path.joinpath('seqs.tar.gz').exists():
                try:
                    result = FitResults.from_json(json_path=f'seqs/{seq_to_hash[seq]}.json',
                                                  tarfile=self.result_path.joinpath('seqs.tar.gz'))
                except:
                    result = FitResults.from_json(json_path=f'results/seqs/{seq_to_hash[seq]}.json',
                                                  tarfile=self.result_path.joinpath('seqs.tar.gz'))

        if result.data.x_data is None and self.data.y_dataframe is not None:
            # add from data attribute
            result.data.x_data = self.data.x_data
            result.data.y_data = self.data.y_dataframe.loc[seq]

        return result

    # def get_record(self, seqs):
    #     """Get record for given seqs"""
    #     if isinstance(seqs, str):
    #         if self.large_dataset:
    #             results = self.get_FitResult(seqs)
    #             return {'bootstrap': results.uncertainty.records, 'convergence': results.convergence.records}
    #         else:
    #             record = {'bootstrap': None, 'convergence': None}
    #             if self._bs_record is not None:
    #                 record['bootstrap'] = self._bs_record[seqs]
    #             if self._conv_record is not None:
    #                 record['convergence'] = self._conv_record[seqs]
    #             return record
    #     else:
    #         if self.large_dataset:
    #             results = {seq: self.get_FitResult(seq) for seq in seqs}
    #             return {seq: {'bootstrap': seq.uncertainty.records, 'convergence': seq.convergence.records} for seq in results}
    #         else:
    #             record = {seq: {'bootstrap': None, 'convergence': None} for seq in seqs}
    #             if self._bs_record is not None:
    #                 for seq in seqs:
    #                     record[seq]['bootstrap'] = self._bs_record[seq]
    #             if self._conv_record is not None:
    #                 for seq in seqs:
    #                     record[seq]['convergence'] = self._conv_record[seq]
    #             return record

    def bs_record(self, seqs=None):
        """Retrieve bootstrap records"""

        if self._bs_record is None:
            return None
        if seqs is None:
            return self._bs_record
        else:
            if isinstance(seqs, str):
                if self.large_dataset:
                    return self.get_FitResult(seqs).uncertainty.records
                else:
                    return self._bs_record[seqs]
            else:
                if self.large_dataset:
                    return {seq: self.get_FitResult(seq).uncertainty.records for seq in seqs}
                else:
                    return {seq: self._bs_record[seq] for seq in seqs}

    def conv_record(self, seqs=None):
        """Retrieve convergence records"""

        if self._conv_record is None:
            return None
        if seqs is None:
            return self._conv_record
        else:
            if isinstance(seqs, str):
                if self.large_dataset:
                    return self.get_FitResult(seqs).convergence.records
                else:
                    return self._bs_record[seqs]
            else:
                if self.large_dataset:
                    return {seq: self.get_FitResult(seq).convergence.records for seq in seqs}
                else:
                    return {seq: self._conv_record[seq] for seq in seqs}

    def summary_to_csv(self, path):
        """Save summary table as csv file"""
        self.summary.to_csv(path)

    def to_pickle(self, output_dir, bs_record=True, conv_record=True):
        """Save fitting results as a single pickled dict, suitable for small dataset.
        For large dataset `to_json` is preferred

        Args:
             output_dir (str): path to saved results, should have suffix of ``.pkl``
             bs_record (bool): if output bs_record, default True
             conv_record (bool): if output conv_record, default True
        """
        check_dir(Path(output_dir).parent)
        data_to_dump = {'summary': self.summary}
        if bs_record:
            bs_record = self.bs_record()
            if isinstance(bs_record, dict):
                # check is type 1
                data_to_dump['bs_record'] = bs_record
            else:
                logging.error('bs_record is not a loaded dict of pd.DataFrame', error_type=TypeError)

        if conv_record:
            conv_record = self.conv_record()
            if isinstance(conv_record, dict):
                # check is type 1
                data_to_dump['conv_record'] = conv_record
            else:
                logging.error('conv_record is not a loaded dict of pd.DataFrame', error_type=TypeError)
        dump_pickle(obj=data_to_dump, path=output_dir)

    @classmethod
    def from_pickle(cls, path_to_pickle, estimator=None, model=None, x_data=None, y_dataframe=None, sigma=None):
        """Create a `BatchFitResults` instance with results loaded from pickle
        Notice:
            this could take a very long time if the pickled file is large, suggest to use to_json for large dataset
        """
        result = cls(estimator=estimator, model=model, x_data=x_data, y_dataframe=y_dataframe, sigma=sigma)
        pkl = read_pickle(path_to_pickle)
        result.summary = pkl['summary']
        if 'bs_record' in pkl.keys():
            result._bs_record = pkl['bs_record']
        if 'conv_record' in pkl.keys():
            result._conv_record = pkl['conv_record']
        result.result_path = path_to_pickle
        result.large_dataset = False
        return result

    def to_json(self, output_dir):
        """Save results in json format, with the structure of
         |output_dir/
             |- summary.csv
             |- seqs
                 |- seq1.json
                 |- seq2.json
                  ...
        Notes:
            Bootstrap and convergence records should already be streamed as separate JSON files under /seqs/

        Args:
             output_dir (str): path of folder to save results
        """
        check_dir(output_dir)
        check_dir(f'{output_dir}/seqs/')
        self.summary.to_csv(f'{output_dir}/summary.csv')
        self.large_dataset = True

    @classmethod
    def from_json(cls, path_to_folder, estimator=None, model=None, x_data=None, y_dataframe=None, sigma=None):
        """Load results from folder of results with json format"""
        result = cls(estimator=estimator, model=model, x_data=x_data, y_dataframe=y_dataframe, sigma=sigma)
        path_to_folder = Path(path_to_folder)
        info(f'loading data from {str(path_to_folder)}...')
        if path_to_folder.joinpath('summary.csv').exists():
            result.summary = pd.read_csv(path_to_folder.joinpath('summary.csv'), index_col=0)
        elif path_to_folder.joinpath('summary.json').exists():
            result.summary = pd.read_json(path_to_folder.joinpath('summary.json'))
            result.summary.index.name = 'seq'

        if path_to_folder.joinpath('seqs').exists():
            info("'seqs' folder found" )
            seq_to_hash = read_json(path_to_folder.joinpath('seqs', 'seq_to_hash.json'))
        elif path_to_folder.joinpath('seqs.tar.gz').exists():
            # for previous saving formats
            import tarfile
            info("'seqs.tar.gz' found" )
            with tarfile.open(path_to_folder.joinpath('seqs.tar.gz'), mode='r:gz') as tf:
                import json
                try:
                    seq_to_hash = json.load(tf.extractfile('seqs/seq_to_hash.json'))
                except:
                    seq_to_hash = json.load(tf.extractfile('results/seqs/seq_to_hash.json'))
        else:
            logging.warning("'seqs' folder or 'seqs.tar.gz' not found - no individual sequence fitting results loaded")
            seq_to_hash = None

        result._bs_record = seq_to_hash
        result._conv_record = seq_to_hash
        result.result_path = Path(path_to_folder)
        result.large_dataset = True
        return result
    
    @classmethod
    def load_result(cls, result_path, estimator=None):
        if Path(result_path).is_file():
            return cls.from_pickle(path_to_pickle=result_path, estimator=estimator)
        else:
            return cls.from_json(path_to_folder=result_path, estimator=estimator)

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


def _read_seq_json(json_path):
    """Read single fitting results from json file and return a summarized pd.Series"""
    fit_res = FitResults.from_json(json_path)
    return fit_res.to_series()


def _read_work_fn(seq):
    """Work function to read JSON results for each sequence"""

    res = _read_seq_json(seq[1])
    res.name = seq[0]
    return res
