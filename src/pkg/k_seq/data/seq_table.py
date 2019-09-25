"""This module contains methods for data preprocessing from count files to ``CountFile`` for estimator
TODO:
  - use sparse matrix, to deprecate SeqSampleSet
  - write output function for each class as JSON file
"""

from ..utility.func_tools import DictToAttr


class Metadata(DictToAttr):

    def __init__(self, attr_dict):
        self.dataset = None
        self.logger = None
        self.samples = None
        self.sequences = None
        super().__init__(attr_dict)


class Grouper(DictToAttr):

    def __init__(self, attr_dict):
        super().__init__(attr_dict=attr_dict)

    def get_table(self, group):
        """todo: implement the methods to directly get table"""
        pass


class SeqTable(object):
    """This class contains the dataset of valid sequences extracted and aligned from a list of ``SeqSampleSet``
    """

    def __init__(self, data_mtx, data_unit='count',
                 seq_list=None, sample_list=None, sample_grouper=None,
                 x_unit=None, note=None, silent=True):

        """
        todo:
          - use hidden variable and seq/sample list for masking count table, amount table, dna amount
        Initialize from a ``SeqSampleSet`` instance
        Find all valid sequences that occur at least once in any 'input' sample and once in any 'reacted' sample
        """

        from ..utility.log import Logger
        from ..utility import allowed_units
        import numpy as np
        import pandas as pd

        allowed_data_unit_mapper = {'count': 'count',
                                    'counts': 'count',
                                    'read': 'count',
                                    'reads': 'count',
                                    'amount': 'amount'}.update({unit:unit for unit in allowed_units})
        if data_unit.lower() not in allowed_data_unit_mapper.keys():
            raise ValueError('Unknown data_type, should be in {}'.format(allowed_data_unit_mapper.keys()))

        from datetime import datetime
        self.metadata = Metadata({
            'dataset': {
                'time': datetime.now(),
                'data_unit': allowed_data_unit_mapper[data_unit],
                'note': note,
                'x_unit': x_unit
            },
            'logger': Logger(silent=silent),
            'samples': None,
            'sequences': None
        })

        self.metadata.logger.add('SeqTable created')

        if isinstance(data_mtx, pd.DataFrame):
            if not data_mtx.apply(pd.api.types.is_sparse).all():
                # convert to sparse data
                if self.metadata.dataset['data_unit'] in ['count']:
                    dtype = pd.SparseDtype('int', fill_value=0)
                else:
                    dtype = pd.SparseDtype('float', fill_value=0.0)
                self._raw_table = data_mtx.astype(dtype)
            else:
                self._raw_table = data_mtx

            self.seq_list = self._raw_table.index.to_series()
            self.sample_list = self._raw_table.columns.to_series()
            if sample_list is not None:
                if isinstance(sample_list, dict):
                    self.metadata.samples = sample_list
                elif isinstance(sample_list, list):
                    pass
                else:
                    raise TypeError('sample_list needs to be a list of sample names or a dictionary of sample metadata')

            if seq_list is not None:
                if isinstance(seq_list, dict):
                    self.metadata.sequences = seq_list
                elif isinstance(seq_list, list):
                    pass
                else:
                    raise TypeError('seq_list needs to be a list of sample names or a dictionary of sample metadata')
            self.metadata.logger.add('Data value imported from Pandas DataFrame, dtype={}'.format(
                self.metadata.dataset['dtype']
            ))
        elif isinstance(data_mtx, np.ndarray):
            if (seq_list is None) or (sample_list is None):
                raise ValueError('seq_list and sample_list must be indicated if using Numpy array')
            else:
                if isinstance(sample_list, dict):
                    self.metadata.samples = sample_list
                    sample_list = sample_list.keys()
                elif isinstance(sample_list, list):
                    pass
                else:
                    raise TypeError('sample_list needs to be a list of sample names or a dict of sample metadata')

                if isinstance(seq_list, dict):
                    self.metadata.sequences = seq_list
                    seq_list = seq_list.keys()
                elif isinstance(seq_list, list):
                    pass
                else:
                    raise TypeError('seq_list needs to be a list of sample names or a dict of sample metadata')
                self._raw_table = pd.DataFrame(pd.SparseArray(data_mtx, fill_value=0),
                                               columns=sample_list,
                                               index=seq_list)
                self.metadata.logger.add('Data value imported from Numpy array, dtype={}'.format(
                    self.metadata.dataset['dtype']
                ))

        if isinstance(sample_grouper, Grouper):
            self.grouper = sample_grouper
        else:
            self.grouper = Grouper(sample_grouper)

        # from .visualizer import seq_occurrence_plot, rep_variability_plot
        # from ..utility.func_tools import FuncToMethod
        # self.visualizer = FuncToMethod(obj=self,
        #                                functions=[
        #                                    seq_occurrence_plot,
        #                                    rep_variability_plot
        #                                ])

    @property
    def table(self):
        return self._raw_table.loc[self.seq_list][self.sample_list]

    def add_norm_table(self, norm_fn, table_name, axis=0):
        self.__dict__[table_name] = self.table.apply(norm_fn, axis=axis)

    def filter_value(self, filter_fn, axis=0, inplace=True):
        "implement the elementwise filter for values, axis=-1 for elementwise, "
        pass

    def filter_axis(self, filter, axis='sample', inplace=True):
        allowed_axis = {
            'sample': 'sample',
            'observation': 'sample',
            1: 'sample',
            'seq': 'sequence',
            'sequences': 'sequence',
            'seqs': 'sequences',
            0: 'sequences'
        }
        if isinstance(axis, str):
            axis = axis.lower()
        if axis not in allowed_axis.keys():
            raise ValueError('Unknown axis, please use sample or sequence')
        else:
            axis = allowed_axis[axis]
        if axis == 'sample':
            handle = self.sample_list.copy()
        else:
            handle = self.seq_list.copy()
        if callable(filter):
            handle = handle[handle.map(filter)]
        elif isinstance(filter, list):
            handle = handle[handle.isin(filter)]
        
        if inplace:
            obj = self
        else:
            from copy import deepcopy
            obj = deepcopy(self)
        if axis == 'sample':
            obj.sample_list = handle
        else:
            obj.seq_list = handle
        if not inplace:
            return obj

    @classmethod
    def from_count_files(cls):
        """todo: implement the method to generate directly from count files"""

    @classmethod
    def from_SeqSampleSet(cls, sample_set,  sample_list=None, black_list=None, seq_list=None, with_spike_in=True,
                          keep_all_seqs=False, use_count=False, note=None):
        """
        todo: move this to SeqSampleSet as to_SeqTable
        Args:

            sample_set (`SeqSampleSet`): valid samples to convert to ``SequenceSet``

            remove_spike_in (`bool`): sequences considered as spike-in will be removed, all number are calculated after
              removal of spike-in

            note (`str`): optional. Additional note to add to the dataset


        Attributes:
`
            metadata (`dict`): dictionary of basic info of dataset:

                - input_seq_num (int): number of unique sequences in all "input" samples

                - reacted_seq_num (int): number of unique sequences in all "reacted" samples

                - valid_seq_num (int): number of valid unqiue sequences that detected in at least one "input" sample
                  and one "reacted" sample

                - remove_spike_in (bool): indicate if spike-in sequences are removed

                - timestamp (time): time the instance created

                - note (str): optional. Additional notes of the dataset

            sample_info (list): a list of dictionaries that preserve original sample information (exclude
                ``SequencingSample.sequences``). Two new item:

                - valid_seq_num (int): number of valid sequences in this sample

                - valid_seq_count (int): total counts of valid sequences in this sample

            count_table (``pandas.DataFrame``): valid sequences and their original counts in valid samples"""
        import pandas as pd

        if sample_list is None:
            sample_list = sample_set.sample_names
        if black_list is not None:
            sample_list = [sample for sample in sample_list if sample not in black_list]

        input_samples = [sample for sample in sample_list if sample_set[sample].sample_type == 'input']
        reacted_samples = [sample for sample in sample_list if sample_set[sample].sample_type == 'reacted']

        if seq_list is None:
            if keep_all_seqs:
                seq_table = sample_set.to_dataframe(samples=sample_list,
                                                    with_spike_in=with_spike_in,
                                                    return_counts=use_count)
            else:
                seq_table = pd.merge(
                    sample_set.to_dataframe(samples=input_samples,
                                            with_spike_in=with_spike_in,
                                            return_counts=use_count),
                    sample_set.to_dataframe(samples=reacted_samples,
                                            with_spike_in=with_spike_in,
                                            return_counts=use_count),
                    left_index=True, right_index=True, how='inner'
                )
        else:
            seq_table = sample_set.to_dataframe(samples=sample_list,
                                                seq_list=seq_list,
                                                with_spike_in=with_spike_in,
                                                return_counts=use_count)

        if note is None:
            note = sample_set.metadata['note']
        # if return raw counts, need to get  dna amount (corrected) as well
        if use_count:

            return cls(count_table=seq_table, input_samples=input_samples, reacted_samples=reacted_samples,
                       metadata=sample_set.metadata.update(
                           {'source_sample_set_summary': sample_set.summary().loc[sample_list]}
                       ), note=note)
        else:
            return cls(amount_table=seq_table, input_samples=input_samples, reacted_samples=reacted_samples,
                       metadata=sample_set.metadata.update(
                           {'source_sample_set_summary': sample_set.summary().loc[sample_list]}
                       ), note=note)


        # find valid sequence set
        self.metadata['remove_spike_in'] = remove_spike_in
        input_set = sample_set.get_samples(sample_id=lambda sample: sample.sample_type == 'input',
                                           with_spike_in=not remove_spike_in)
        reacted_set = sample_set.get_samples(sample_id=lambda sample: sample.sample_type == 'reacted',
                                            with_spike_in=not remove_spike_in)
        valid_set = set(input_set.index) & set(reacted_set.index)
        self.metadata['seq_nums'] = {
            'input_seq_num': input_set.shape[0],
            'reacted_seq_num': reacted_set.shape[0],
            'valid_seq_num': len(valid_set)
        }

        self.count_table_reacted = reacted_set.loc[valid_set]
        self.count_table_input = input_set.loc[valid_set]

        # preserve sample info
        self.sample_info = {}
        for sample in sample_set.sample_set:
            sample_info_dict = sample.__dict__.copy()
            _ = sample_info_dict.pop('visualizer')
            sequences = sample_info_dict.pop('_sequences', None)
            self.sample_info[sample.name] = {
                'valid_seqs_num': len(set(sequences.index.values) & valid_set),
                'valid_seqs_counts': np.sum(sequences.loc[list(set(sequences.index.values) & valid_set)]['counts'])
            }
            self.sample_info[sample.name].update(sample_info_dict)


    def get_reacted_frac(self, input_average='median', black_list=None, inplace=True):
        """Calculate reacted fraction for sequences

        Args:

            input_average ('median' or 'mean'): method to calculate the average amount of input for a sequence

            black_list (list of `str`): optional, list of names of samples to be excluded in calculation

            inplace (bool): add ``reacted_frac_table`` to the attribute of instance if True; return
                ``reacted_frac_table`` if False

        Returns:

            reacted_frac_table (if *inplace* is False)

        Attributes:

            reacted_frac_table (``pandas.DataFrame``): a ``DataFrame`` object containing the reacted fraction of "reacted"
                samples for all valid sequences. Extra attributes are added to the ``DataFrame``:

                - input_avg_type ('median' or 'mean'): method used to calculate input average

                - col_x_values (list of float): time points or concentration points values for "reacted" samples

                - input_avg (numpy.Array): 1D array containing the average values on input for valid sequences
        """

        if not black_list:
            black_list = []
        col_to_use = [col_name for col_name in self.count_table_reacted.columns if col_name not in black_list]
        reacted_frac_table = self.count_table_reacted[col_to_use]
        reacted_frac_table = reacted_frac_table.apply(
            lambda sample: sample/self.sample_info[sample.name]['total_counts'] * self.sample_info[sample.name]['quant_factor'],
            axis=0
        )
        self.metadata['input_avg_type'] = input_average
        input_amount = self.count_table_input.loc[reacted_frac_table.index]
        input_amount = input_amount.apply(
            lambda sample: sample / self.sample_info[sample.name]['total_counts'] * self.sample_info[sample.name]['quant_factor'],
            axis=0
        )
        if input_average == 'median':
            input_amount_avg = input_amount.median(axis=1)
        elif input_average == 'mean':
            input_amount_avg = input_amount.median(axis=1)
        else:
            raise Exception("Error: input_average should be 'median' or 'mean'")
        reacted_frac_table = reacted_frac_table.divide(input_amount_avg, axis=0)
        if inplace:
            self.reacted_frac_table = reacted_frac_table
            self.logger.add('reacted_frac_tabled added using {} as input average'.format(input_average))
        else:
            return reacted_frac_table

    @property
    def x_values(self):
        """Return x values corresponding to each column in `reacted_frac_table` (or `count_table_reacted`)
        as pd.Series
        """
        import pandas as pd

        if hasattr(self, 'reacted_frac_table'):
            table = self.reacted_frac_table
        else:
            table = self.count_table_reacted
        return pd.Series(data=[self.sample_info[sample]['x_value'] for sample in table.columns],
                         index=table.columns)

    @property
    def seq_info(self):
        import pandas as pd
        import numpy as np

        seq_info = pd.DataFrame(index=self.count_table_input.index)
        seq_info['occurred_in_inputs'] = pd.Series(np.sum(self.count_table_input > 0, axis=1))
        seq_info['occurred_in_reacted'] = pd.Series(np.sum(self.count_table_reacted > 0, axis=1))
        get_rel_abun = lambda series: series/series.sum()
        seq_info['avg_rel_abun_in_inputs'] = pd.Series(
            self.count_table_input.apply(get_rel_abun, axis=0).mean(axis=1)
        )
        seq_info['avg_rel_abun_in_reacted'] = pd.Series(
            self.count_table_reacted.apply(get_rel_abun, axis=0).mean(axis=1)
        )
        return seq_info.sort_values(by='avg_rel_abun_in_inputs', ascending=False)


    def add_fitting(self, model, seq_to_fit=None, weights=None, bounds=None,
                    bootstrap_depth=0, bs_return_size=None,
                    resample_pct_res=False, missing_data_as_zero=False, random_init=True, metrics=None):
        """
        Add a `k_seq.estimator.BatchFitting` instance to SeqTable for estimator
        Args:
            model (`callable`): the model to fit
            seq_to_fit (list of `str`): optional. All the sequences will be fit if None
            weights (list of `float`): optional. If assign different weights in the estimator for sample points.
            bounds (k by 2 list of `float`): optional. If set bounds for each parameters to fit
            bootstrap_depth (`int`): optional. Number of bootstrap to perform. No bootstrap if None
            bs_return_size (`int`): optional. If only keep part of the bootstrap results for memory
            resample_pct_res (`bool`):
            missing_data_as_zero (`bool`): If treat missing value as zero. Default False
            random_init (`bool`): If use random initialization between [0, 1] in optimization for each parameter, default True
            metrics (`dict` of `callable`): optional. If calculate other metrics from estimated parameter. Has form
              {
                metric_name: callable_to_cal_metric_from_pd.Series
            }

        """
        from ..estimator.least_square import BatchFitting
        if seq_to_fit is None:
            seq_to_fit = None
        if weights is None:
            weights = None
        if bounds is None:
            bounds = None
        if bs_return_size is None:
            bs_return_size = None
        if metrics is None:
            metrics = None
        self.fitting = BatchFitting.from_SeqTable(
            seq_table=self,
            model=model,
            seq_to_fit=seq_to_fit,
            weights=weights,
            bounds=bounds,
            bootstrap_depth=bootstrap_depth,
            bs_return_size=bs_return_size,
            resample_pct_res=resample_pct_res,
            missing_data_as_zero=missing_data_as_zero,
            random_init=random_init,
            metrics=metrics
        )
        self.logger.add('BatchFitting fitter added')

    @classmethod
    def load_count_files(cls, folder_path, x_values, file_pattern=None, name_pattern=None, file_list=None, black_list=None,
                         x_unit=None, dna_unit=None,
                         spike_in_seq=None, spike_in_amount=None, spike_in_dia=None, ):
        """To accommodate for exploratery needs, this will be a wrapper for `count_file.CountFileSet"""
        note = f'Loaded from {folder_path}'




    def save_as_dill(self, dirc):
        import dill
        with open(dirc, 'w') as handle:
            handle.write(dill.dumps(self))

    @staticmethod
    def load_from_dill(dirc):
        import dill
        with open(dirc) as handle:
            return dill.loads(handle.readline())


class SeqFilter:

    class Filter:
        def __init__(self, func, value):
            self.func = func
            self.value = value

    def __init__(self, seq_table, seq_length_range=None, max_edit_dist_to_seqs=None,
                 min_occur_input=None, min_occur_reacted=None,
                 min_counts_input=None, min_counts_reacted=None,
                 min_rel_abun_input=None, min_rel_abun_reacted=None):
        """
        Filter object with some built-in filter options

        Use `SeqFilter.filter_fn` to get the `callable` function

        Use `SeqFilter.seq_to_keep` to get a list of sequences passed the filters

        Args:
            seq_table (`SeqTable`): the `SeqTable` instance to apply filters on
            seq_length_range ([min, max]): only keep sequences within range [min, max]
            max_edit_dist_to_seqs (`int`):
            min_counts_input (`int`):
            min_counts_reacted (`int`):
            min_rel_abun_input (`float`): relative abundance is only based on valid sequences
            min_rel_abun_reacted (`float`): relative abundance is only based on valid sequences
        """

        import numpy as np
        import pandas as pd

        self.seq_table = seq_table

        if seq_length_range is not None:
            self.seq_length_range = self.Filter(
                func=lambda seq: seq_length_range[0] <= len(seq) <= seq_length_range[1],
                value = seq_length_range)

        if max_edit_dist_to_seqs is not None:
            if isinstance(max_edit_dist_to_seqs, list) or isinstance(max_edit_dist_to_seqs, tuple):
                max_edit_dist_to_seqs = {seq[0]: int(seq[1]) for seq in max_edit_dist_to_seqs}

            def edit_dist_filter_fn(seq):
                import Levenshtein
                flag = True
                for target, max_dist in max_edit_dist_to_seqs.items():
                    flag = flag and Levenshtein.distance(seq, target) <= max_dist
                return flag

            self.max_edit_dist_to_seqs = self.Filter(
                func=edit_dist_filter_fn,
                value=max_edit_dist_to_seqs
            )

        if min_occur_input is not None:
            self.min_occur_input = self.Filter(
                func=lambda seq: np.sum(self.seq_table.count_table_input.loc[seq] > 0) >= min_occur_input,
                value=min_occur_input
            )
        if min_occur_reacted is not None:
            self.min_occur_reacted = self.Filter(
                func=lambda seq: np.sum(self.seq_table.count_table_reacted.loc[seq] > 0) >= min_occur_reacted,
                value=min_occur_reacted
            )


        if min_counts_input is not None:
            self.min_counts_input = self.Filter(
                func=lambda seq: self.seq_table.count_table_input.loc[seq].mean() >= min_counts_input,
                value=min_counts_input
            )

        if min_counts_reacted is not None:
            self.min_counts_reacted = self.Filter(
                func=lambda seq: self.seq_table.count_table_reacted.loc[seq].mean() >= min_counts_reacted,
                value=min_counts_reacted
            )

        if min_rel_abun_input is not None:
            self.min_rel_abun_input = self.Filter(
                func=lambda seq: (self.seq_table.count_table_input.loc[seq]/self.seq_table.count_table_input.sum(axis=0)).mean() >= min_rel_abun_input,
                value=min_rel_abun_input
            )

        if min_rel_abun_reacted is not None:
            self.min_rel_abun_reacted = self.Filter(
                func=lambda seq: (self.seq_table.count_table_reacted.loc[seq]/self.seq_table.count_table_reacted.sum(axis=0)).mean() >= min_rel_abun_reacted,
                value=min_rel_abun_reacted
            )

    def print_filters(self):
        print('Following filter added:')
        for filter,content in self.__dict__.items():
            if isinstance(content, self.Filter):
                print('\t{}:{}'.format(filter, content.value))

    def apply_filters(self):
        seq_to_keep = self.seq_table.count_table_reacted.index
        self.seq_to_keep = seq_to_keep[seq_to_keep.map(self.filter_fn)]

    @property
    def filter_fn(self):
        def _filter_fn(seq):
            import numpy as np
            flags = [filter.func(seq) for filter in self.__dict__.values()
                     if isinstance(filter, self.Filter)]
            return np.all(flags)
        return _filter_fn



