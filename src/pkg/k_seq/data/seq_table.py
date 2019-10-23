"""This module contains methods for data preprocessing from count files to ``CountFile`` for estimator
TODO:
  - write output function for each class as JSON file
  - Formalized filter and normalizer
"""

from ..utility.func_tools import DictToAttr


class Metadata(DictToAttr):

    def __init__(self, attr_dict):
        self.dataset = None
        self.logger = None
        self.samples = None
        self.sequences = None
        super().__init__(attr_dict)


def slice_table(table, axis, remove_zero, keys=None, mask=None):
    if axis == 0:
        if keys is not None:
            sub_table = table.loc[keys]
        if mask is not None:
            sub_table = table.loc[mask]
        if remove_zero:
            return sub_table.loc[:, (sub_table != 0).any(axis=0)]
        else:
            return sub_table
    else:
        if keys is not None:
            sub_table = table[keys]
        if mask is not None:
            sub_table = table[mask]
        if remove_zero:
            return sub_table.loc[(sub_table != 0).any(axis=1)]
        else:
            return sub_table


class SeqTable(object):
    """This class contains the dataset of valid sequences extracted and aligned from a list of ``SeqSampleSet``
    """

    def __init__(self, data_mtx, data_unit='count',
                 seq_list=None, sample_list=None, grouper=None,
                 seq_metadata=None, sample_metadata=None,
                 x_values=None, x_unit=None, note=None, dataset_metadata=None, silent=True):
        """
        todo:
          - use hidden variable and seq/sample list for masking count table, amount table, dna amount
          - separate spike-in from sample metadata, make it optional
        Initialize from a ``SeqSampleSet`` instance
        Find all valid sequences that occur at least once in any 'input' sample and once in any 'reacted' sample
        """

        from ..utility.log import Logger
        import numpy as np
        import pandas as pd

        # check input data unit type
        allowed_data_unit_mapper = {'count': 'count',
                                    'counts': 'count',
                                    'read': 'count',
                                    'reads': 'count',
                                    'amount': 'amount'}
        from ..utility import allowed_units
        allowed_data_unit_mapper.update({unit:unit for unit in allowed_units})
        if data_unit.lower() not in allowed_data_unit_mapper.keys():
            raise ValueError('Unknown data_type, should be in {}'.format(allowed_data_unit_mapper.keys()))

        # initialize metadata
        from datetime import datetime
        self.metadata = Metadata({
            'dataset': DictToAttr({
                'time': datetime.now(),
                'data_unit': allowed_data_unit_mapper[data_unit],
                'note': note,
                'x_unit': x_unit
            }),
            'logger': Logger(silent=silent),
        })
        if dataset_metadata is not None:
            self.metadata.dataset.add(dataset_metadata)
        if sample_metadata is not None:
            self.metadata.samples = DictToAttr(sample_metadata)
        if seq_metadata is not None:
            self.metadata.sequences = DictToAttr(seq_metadata)
        self.metadata.logger.add('SeqTable created')

        # import table
        self.table = None
        if isinstance(data_mtx, pd.DataFrame):
            if not data_mtx.apply(pd.api.types.is_sparse).all():
                # not sparse, convert to sparse data
                if self.metadata.dataset['data_unit'] in ['count']:
                    dtype = pd.SparseDtype('int', fill_value=0)
                else:
                    dtype = pd.SparseDtype('float', fill_value=0.0)
                self._raw_table = data_mtx.astype(dtype)
                self.metadata.logger.add('Import type pd.Dataframe, not sparse, convert to sparse table')
            else:
                self._raw_table = data_mtx
                self.metadata.logger.add('Import type pd.Dataframe, sparse')
            self.seq_list = self._raw_table.index.to_series()
            self.sample_list = self._raw_table.columns.to_series()

            if sample_list is not None:
                # if sample list/seq list is also provided, it needs to be the subset of the Dataframe samples
                if False in list(pd.Series(sample_list).isin(list(self._sample_list))):
                    raise ValueError('Some samples are not found in data_mtx')
                else:
                    self.sample_list = sample_list

            if seq_list is not None:
                if False in list(pd.Series(seq_list).isin(list(self._seq_list))):
                    raise ValueError('Some seq are not found in data_mtx')
                else:
                    self.seq_list = seq_list
            self.metadata.logger.add('Data value imported from Pandas DataFrame, dtype={}'.format(
                self.metadata.dataset['data_unit']
            ))
        elif isinstance(data_mtx, np.ndarray):
            if (seq_list is None) or (sample_list is None):
                raise ValueError('seq_list and sample_list must be indicated if using Numpy array')
            else:
                if len(sample_list) != data_mtx.shape[1]:
                    raise ValueError('Length of sample_list does not match with data_mtx')
                else:
                    self.sample_list = sample_list

                if len(seq_list) != data_mtx.shape[0]:
                    raise ValueError('Length of seq_list does not match with data_mtx')
                else:
                    self.seq_list = seq_list

                self._raw_table = pd.DataFrame(pd.SparseArray(data_mtx, fill_value=0),
                                               columns=sample_list,
                                               index=seq_list)
                self.metadata.logger.add('Data value imported from Numpy array, dtype={}'.format(
                    self.metadata.dataset['data_unit']
                ))
        else:
            raise TypeError('data_mtx type is not supported')

        self.table = self._raw_table.loc[self.seq_list][self.sample_list]

        if x_values is None:
            self.x_values = None
        elif isinstance(x_values, dict):
            self.x_values = pd.Series(x_values)
        elif isinstance(x_values, (list, np.ndarray)):
            self.x_values = pd.Series(x_values, index=self.table.columns)
        else:
            raise TypeError('Unknown type for x_values')

        # import grouper
        from .grouper import Grouper
        if grouper is not None:
            if isinstance(grouper, Grouper):
                self.grouper = grouper
            else:
                self.grouper = Grouper(grouper, target=self.table)
        else:
            self.grouper = None

        # from .visualizer import seq_occurrence_plot, rep_variability_plot
        # from ..utility.func_tools import FuncToMethod
        # self.visualizer = FuncToMethod(obj=self,
        #                                functions=[
        #                                    seq_occurrence_plot,
        #                                    rep_variability_plot
        #                                ])

    # @property
    # def table(self):
    #     """Deprecated due to poor slicing performance on larget table"""
    #     return self._raw_table.loc[self.seq_list][self.sample_list]

    @property
    def sample_list(self):
        return self._sample_list

    @sample_list.setter
    def sample_list(self, sample_list):
        if hasattr(self, '_sample_list'):
            if set(sample_list) != set(self._sample_list):
                self._sample_list = sample_list
                self.table = self._raw_table.loc[self.seq_list][sample_list]
        else:
            self._sample_list = sample_list

    @property
    def seq_list(self):
        return self._seq_list

    @seq_list.setter
    def seq_list(self, seq_list):
        if hasattr(self, '_seq_list'):
            if set(seq_list) != set(self._seq_list):
                self._seq_list = seq_list
                self.table = self._raw_table.loc[seq_list][self.sample_list]
        else:
            self._seq_list = seq_list

#     def add_norm_table(self, norm_fn, table_name, axis=0):
#         setattr(self, table_name, self.table.apply(norm_fn, axis=axis))
#
#     def filter_value(self, filter_fn, axis=0, inplace=True):
#         "implement the elementwise filter for values, axis=-1 for elementwise, "
#         pass
#
#     def filter_axis(self, filter, axis='sample', inplace=True):
#         allowed_axis = {
#             'sample': 'sample',
#             'observation': 'sample',
#             1: 'sample',
#             'seq': 'sequence',
#             'sequences': 'sequence',
#             'seqs': 'sequences',
#             0: 'sequences'
#         }
#         if isinstance(axis, str):
#             axis = axis.lower()
#         if axis not in allowed_axis.keys():
#             raise ValueError('Unknown axis, please use sample or sequence')
#         else:
#             axis = allowed_axis[axis]
#         if axis == 'sample':
#             handle = self.sample_list.copy()
#         else:
#             handle = self.seq_list.copy()
#         if callable(filter):
#             handle = handle[handle.map(filter)]
#         elif isinstance(filter, list):
#             handle = handle[handle.isin(filter)]
#
#         if inplace:
#             obj = self
#         else:
#             from copy import deepcopy
#             obj = deepcopy(self)
#         if axis == 'sample':
#             obj.sample_list = handle
#         else:
#             obj.seq_list = handle
#         if not inplace:
#             return obj

    @classmethod
    def from_count_files(cls,
                         file_root, file_list=None, pattern_filter=None, black_list=None, name_pattern=None, sort_by=None,
                         x_values=None, x_unit=None, input_sample_name=None, sample_metadata=None, note=None,
                         silent=True, dry_run=False, **kwargs):
        """todo: implement the method to generate directly from count files"""
        from ..utility.file_tools import get_file_list, extract_metadata
        import numpy as np
        import pandas as pd

        # parse file metadata
        file_list = get_file_list(file_root=file_root, file_list=file_list,
                                  pattern=pattern_filter, black_list=black_list, full_path=True)
        if name_pattern is None:
            samples = {file.name: {'file_path': str(file), 'name':file.name} for file in file_list}
        else:
            samples = {}
            for file in file_list:
                f_meta = extract_metadata(target=file.name, pattern=name_pattern)
                samples[f_meta['name']] = {**f_meta, **{'file_path': str(file)}}
        if sample_metadata is not None:
            for file_name, f_meta in sample_metadata.items():
                samples[file_name].udpate(f_meta)

        # sort file order if applicable
        sample_names = list(samples.keys())
        if sort_by is not None:
            if isinstance(sort_by, str):
                def sort_fn(sample_name):
                    return samples[sample_name].get(sort_by, np.nan)
            elif callable(sort_by):
                sort_fn = sort_by
            else:
                raise TypeError('Unknown sort_by format')
            sample_names = sorted(sample_names, key=sort_fn)

        if dry_run:
            return pd.DataFrame(samples)[sample_names].transpose()

        from ..data.count_file import read_count_file
        data_mtx = {sample: read_count_file(file_path=samples[sample]['file_path'], as_dict=True)[2]
                    for sample in sample_names}
        data_mtx = pd.DataFrame.from_dict(data_mtx).fillna(0, inplace=False).astype(pd.SparseDtype(dtype='int'))
        if input_sample_name is not None:
            grouper = {'input': [name for name in sample_names if name in input_sample_name],
                       'reacted': [name for name in sample_names if name not in input_sample_name]}
        else:
            grouper = None

        seq_table = cls(data_mtx, data_unit='count', grouper=grouper, sample_metadata=sample_metadata,
                        x_values=x_values, x_unit=x_unit, note=note, silent=silent)

        if 'spike_in_seq' in kwargs.keys():
            seq_table.add_spike_in(**kwargs)

        # todo: add total dna amount normalizer if applicable
        if 'dna_amount' in kwargs.keys():
            seq_table.add_total_dna_amount(**kwargs)

        return seq_table

    def add_spike_in(self, spike_in_seq, spike_in_amount, radius=2, dna_unit=None, black_list=None):
        """Add spike in """
        from .transform import SpikeInNormalizer
        setattr(self, 'spike_in',
                SpikeInNormalizer(target=self, spike_in_seq=spike_in_seq, spike_in_amount=spike_in_amount,
                                  radius=radius, unit=dna_unit, blacklist=black_list))

    def add_total_dna_amount(self, dna_amount, dna_unit=None):
        """todo: add total DNA normalizer"""
        pass

    def sample_overview(self):
        import numpy as np
        import pandas as pd

        def get_sample_info(self, sample_name):
            info = {'name': sample_name}
            try:
                if sample_name in self.grouper.input:
                    info['sample type'] = 'input'
                elif sample_name in self.grouper.reacted:
                    info['sample type'] = 'reacted'
                else:
                    pass
            except AttributeError:
                pass
            try:
                info['x value'] = self.x_values[sample_name]
            except AttributeError:
                pass
            try:
                info['unique seqs'] = np.sum((self.table[sample_name] > 0).sparse.to_dense())
            except KeyError:
                pass
            try:
                info['total counts'] = np.sum(self.table[sample_name].sparse.to_dense())
            except KeyError:
                pass
            try:
                info[f"dna amount (from spike-in{'' if self.spike_in is None else ', ' + self.spike_in.unit})"] = \
                    self.spike_in.norm_factor[sample_name] * info['total counts']
                info['spike-in rad'] = self.spike_in.radius
                info['spike-in pct'] = np.sum((self.table[sample_name][self.spike_in.spike_in_members]).sparse.to_dense()) /\
                                       info['total counts']
            except AttributeError:
                pass
            except KeyError:
                pass
            return info

        sample_info = {sample_name: get_sample_info(self=self, sample_name=sample_name)
                        for sample_name in self.sample_list}
        return pd.DataFrame.from_dict(sample_info, orient='index')

    def seq_overview(self, target=None):
        """
        todo: finish seq_table
        columns to include:
        - length
        - occurrence / occurrence in input/reacted
        - rel abun / mean rel in input/reacted

        Returns:
            A `pd.DataFrame` show the summary for sequences

        """
        if target is None:
            target = self.table
        elif isinstance(target, str):
            try:
                target = getattr(self, target)
            except AttributeError:
                raise AttributeError(f'Table {target} not found')
        else:
            target = target
        pass

    @classmethod
    def load_default_dataset(cls, dataset, from_count_file=False):
        """Load default dataset
        Available dataset:
          - BYO-doped: 'byo-doped'
          - BYO: not implemented
          - BFO: not implemented
        """
        if dataset.lower() in ['byo_doped', 'byo-doped', 'doped']:
            return cls._load_byo_doped(from_count_file=from_count_file)
        else:
            raise NotImplementedError(f'Dataset {dataset} is not implemented')

    @classmethod
    def _load_byo_doped(cls, from_count_file=False):
        """todo: add dataset description"""
        BYO_DOPED_PKL = '/mnt/storage/projects/k-seq/datasets/byo_doped.pkl'
        BYO_DOPED_COUNT_FILE = '/mnt/storage/projects/k-seq/input/byo_doped/counts'
        if from_count_file:
            import numpy as np

            print('Generate SeqTable instance for BYO-doped pool...')
            print(f'Importing from {BYO_DOPED_COUNT_FILE}...this could take a couple of minutes...')

            byo_doped = cls.from_count_files(
                file_root=BYO_DOPED_COUNT_FILE,
                pattern_filter='counts-',
                name_pattern='counts-d-[{byo}{exp_rep}].txt',
                dry_run=False,
                sort_by='name',
                x_values=np.concatenate((
                    np.repeat([1250, 250, 50, 10, 2], repeats=3) * 1e-6,
                    np.array([np.nan])), axis=0
                ),
                x_unit='mol',
                spike_in_seq='AAAAACAAAAACAAAAACAAA',
                spike_in_amount=np.concatenate((
                    np.repeat([2, 2, 1, 0.2, .04], repeats=3),
                    np.array([10])), axis=0    # input pool sequenced is 3-times of actual initial pool
                ),
                radius=4,
                dna_unit='ng',
                input_sample_name=['R0']
            )

            # Add standard filters
            from . import filters
            spike_in_filter = filters.SpikeInFilter(target=byo_doped)  # remove spike-in seqs
            seq_length_filter = filters.SeqLengthFilter(target=byo_doped, min_len=21, max_len=21) # remove
            singleton_filter = filters.SingletonFilter(target=byo_doped)

            byo_doped.filtered_table = singleton_filter.get_filtered_table(
                target=seq_length_filter.get_filtered_table(
                    target=spike_in_filter.get_filtered_table()
                )
            )

            # Add replicate grouper
            byo_doped.grouper.add({'byo': {
                1250: ['A1', 'A2', 'A3'],
                250: ['B1', 'B2', 'B3'],
                50: ['C1', 'C2', 'C3'],
                10: ['D1', 'D2', 'D3'],
                2: ['E1', 'E2', 'E3']
            }}, target=byo_doped.filtered_table)

            # normalized using spike-in
            byo_doped.abs_amnt_filtered = byo_doped.spike_in.apply(target=byo_doped.filtered_table)

            # calculate reacted faction
            from .transform import ReactedFractionNormalizer
            reacted_frac = ReactedFractionNormalizer(target=byo_doped,
                                                     input_pools=['R0'],
                                                     abs_amnt_table='abs_amnt_filtered',
                                                     reduce_method='median',
                                                     remove_zero=True)
            byo_doped.reacted_frac_filtered = reacted_frac.apply()
            print('Finished!')
        else:
            print(f'Load BYO-doped pool data from pickled record at {BYO_DOPED_PKL}')
            import pickle
            with open(BYO_DOPED_PKL, 'rb') as handle:
                byo_doped = pickle.load(handle)
            print('Imported!')
        # print('Following pre-processing has been applied')
        return byo_doped


    #     #
    #     # @property
    #     # def seq_info(self): todo: revive this
    #     #     import pandas as pd
    #     #     import numpy as np
    #     #
    #     #     seq_info = pd.DataFrame(index=self.count_table_input.index)
    #     #     seq_info['occurred_in_inputs'] = pd.Series(np.sum(self.count_table_input > 0, axis=1))
    #     #     seq_info['occurred_in_reacted'] = pd.Series(np.sum(self.count_table_reacted > 0, axis=1))
    #     #     get_rel_abun = lambda series: series/series.sum()
    #     #     seq_info['avg_rel_abun_in_inputs'] = pd.Series(
    #     #         self.count_table_input.apply(get_rel_abun, axis=0).mean(axis=1)
    #     #     )
    #     #     seq_info['avg_rel_abun_in_reacted'] = pd.Series(
    #     #         self.count_table_reacted.apply(get_rel_abun, axis=0).mean(axis=1)
    #     #     )
    #     #     return seq_info.sort_values(by='avg_rel_abun_in_inputs', ascending=False)


#     @classmethod
#     def from_SeqSampleSet(cls, sample_set,  sample_list=None, black_list=None, seq_list=None, with_spike_in=True,
#                           keep_all_seqs=False, use_count=False, note=None):
#         """
#         todo: move this to SeqSampleSet as to_SeqTable
#         Args:
#
#             sample_set (`SeqSampleSet`): valid samples to convert to ``SequenceSet``
#
#             remove_spike_in (`bool`): sequences considered as spike-in will be removed, all number are calculated after
#               removal of spike-in
#
#             note (`str`): optional. Additional note to add to the dataset
#
#
#         Attributes:
# `
#             metadata (`dict`): dictionary of basic info of dataset:
#
#                 - input_seq_num (int): number of unique sequences in all "input" samples
#
#                 - reacted_seq_num (int): number of unique sequences in all "reacted" samples
#
#                 - valid_seq_num (int): number of valid unqiue sequences that detected in at least one "input" sample
#                   and one "reacted" sample
#
#                 - remove_spike_in (bool): indicate if spike-in sequences are removed
#
#                 - timestamp (time): time the instance created
#
#                 - note (str): optional. Additional notes of the dataset
#
#             sample_info (list): a list of dictionaries that preserve original sample information (exclude
#                 ``SequencingSample.sequences``). Two new item:
#
#                 - valid_seq_num (int): number of valid sequences in this sample
#
#                 - valid_seq_count (int): total counts of valid sequences in this sample
#
#             count_table (``pandas.DataFrame``): valid sequences and their original counts in valid samples"""
#         pass
#
#     # def get_reacted_frac(self, input_average='median', black_list=None, inplace=True):
#     #     """Calculate reacted fraction for sequences
#     #
#     #     Args:
#     #
#     #         input_average ('median' or 'mean'): method to calculate the average amount of input for a sequence
#     #
#     #         black_list (list of `str`): optional, list of names of samples to be excluded in calculation
#     #
#     #         inplace (bool): add ``reacted_frac_table`` to the attribute of instance if True; return
#     #             ``reacted_frac_table`` if False
#     #
#     #     Returns:
#     #
#     #         reacted_frac_table (if *inplace* is False)
#     #
#     #     Attributes:
#     #
#     #         reacted_frac_table (``pandas.DataFrame``): a ``DataFrame`` object containing the reacted fraction of "reacted"
#     #             samples for all valid sequences. Extra attributes are added to the ``DataFrame``:
#     #
#     #             - input_avg_type ('median' or 'mean'): method used to calculate input average
#     #
#     #             - col_x_values (list of float): time points or concentration points values for "reacted" samples
#     #
#     #             - input_avg (numpy.Array): 1D array containing the average values on input for valid sequences
#     #     """
#     #
#     #     if not black_list:
#     #         black_list = []
#     #     col_to_use = [col_name for col_name in self.count_table_reacted.columns if col_name not in black_list]
#     #     reacted_frac_table = self.count_table_reacted[col_to_use]
#     #     reacted_frac_table = reacted_frac_table.apply(
#     #         lambda sample: sample/self.sample_info[sample.name]['total_counts'] * self.sample_info[sample.name]['quant_factor'],
#     #         axis=0
#     #     )
#     #     self.metadata['input_avg_type'] = input_average
#     #     input_amount = self.count_table_input.loc[reacted_frac_table.index]
#     #     input_amount = input_amount.apply(
#     #         lambda sample: sample / self.sample_info[sample.name]['total_counts'] * self.sample_info[sample.name]['quant_factor'],
#     #         axis=0
#     #     )
#     #     if input_average == 'median':
#     #         input_amount_avg = input_amount.median(axis=1)
#     #     elif input_average == 'mean':
#     #         input_amount_avg = input_amount.median(axis=1)
#     #     else:
#     #         raise Exception("Error: input_average should be 'median' or 'mean'")
#     #     reacted_frac_table = reacted_frac_table.divide(input_amount_avg, axis=0)
#     #     if inplace:
#     #         self.reacted_frac_table = reacted_frac_table
#     #         self.logger.add('reacted_frac_tabled added using {} as input average'.format(input_average))
#     #     else:
#     #         return reacted_frac_table
#     #

#     #
#     #
#     # def add_fitting(self, model, seq_to_fit=None, weights=None, bounds=None,
#     #                 bootstrap_depth=0, bs_return_size=None,
#     #                 resample_pct_res=False, missing_data_as_zero=False, random_init=True, metrics=None):
#     #     """
#     #     Add a `k_seq.estimator.BatchFitting` instance to SeqTable for estimator
#     #     Args:
#     #         model (`callable`): the model to fit
#     #         seq_to_fit (list of `str`): optional. All the sequences will be fit if None
#     #         weights (list of `float`): optional. If assign different weights in the estimator for sample points.
#     #         bounds (k by 2 list of `float`): optional. If set bounds for each parameters to fit
#     #         bootstrap_depth (`int`): optional. Number of bootstrap to perform. No bootstrap if None
#     #         bs_return_size (`int`): optional. If only keep part of the bootstrap results for memory
#     #         resample_pct_res (`bool`):
#     #         missing_data_as_zero (`bool`): If treat missing value as zero. Default False
#     #         random_init (`bool`): If use random initialization between [0, 1] in optimization for each parameter, default True
#     #         metrics (`dict` of `callable`): optional. If calculate other metrics from estimated parameter. Has form
#     #           {
#     #             metric_name: callable_to_cal_metric_from_pd.Series
#     #         }
#     #
#     #     """
#     #     from ..estimator.least_square import BatchFitting
#     #     if seq_to_fit is None:
#     #         seq_to_fit = None
#     #     if weights is None:
#     #         weights = None
#     #     if bounds is None:
#     #         bounds = None
#     #     if bs_return_size is None:
#     #         bs_return_size = None
#     #     if metrics is None:
#     #         metrics = None
#     #     self.fitting = BatchFitting.from_SeqTable(
#     #         seq_table=self,
#     #         model=model,
#     #         seq_to_fit=seq_to_fit,
#     #         weights=weights,
#     #         bounds=bounds,
#     #         bootstrap_depth=bootstrap_depth,
#     #         bs_return_size=bs_return_size,
#     #         resample_pct_res=resample_pct_res,
#     #         missing_data_as_zero=missing_data_as_zero,
#     #         random_init=random_init,
#     #         metrics=metrics
#     #     )
#     #     self.logger.add('BatchFitting fitter added')
#     #
#     # @classmethod
#     # def load_count_files(cls, folder_path, x_values, file_pattern=None, name_pattern=None, file_list=None, black_list=None,
#     #                      x_unit=None, dna_unit=None,
#     #                      spike_in_seq=None, spike_in_amount=None, spike_in_dia=None, ):
#     #     """To accommodate for exploratery needs, this will be a wrapper for `count_file.CountFileSet"""
#     #     note = f'Loaded from {folder_path}'
#     #

    def to_pickle(self, path):
        import pickle
        with open(path, 'wb') as handle:
            pickle.dump(self, handle, protocol=-1)

    @staticmethod
    def from_pickle(path):
        import pickle
        with open(path, 'rb') as handle:
            return pickle.load(handle)
