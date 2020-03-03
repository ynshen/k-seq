"""Submodule of `SeqTable`, a rich functions class of table for sequencing manipulation This module contains methods
for data pre-processing from count files to ``CountFile`` for estimator
For absolute quantification, it accepts absolute amount (e.g. measured by qPCR) or reacted fraction
TODO:
  - move table under scope of table
  - refactor plugins to plugins
  - write output function for each class as JSON file
  - Formalized filter and normalizer
"""
import numpy as np
import pandas as pd
from ..utility.func_tools import AttrScope
from ..utility.file_tools import _name_template_example
from ..utility.log import logging, Logger
from doc_helper import DocHelper


_table_doc = DocHelper(
    x_values=('list-like, or dict', 'optional. value for controlled variables. If list-like, should have same length '
                                    'and order as samples; if dict, should have sample names as key'),
    x_unit=('str', "optional. Unit for controlled variable. e.g. 'uM'"),
    input_sample_name=('list of str', 'optional. Indicate input samples (unreacted)'),
    sample_metadata=('dict or pd.DataFrame', 'optional. Extra sample metadata'),
    note=('str', 'Note for dataset')
)


class Table(pd.DataFrame):
    """Subclass of pd.DataFrame with added property and functions for sequencing table

    Additional properties and methods:
        unit (str): unit of entries in this table
        note (str): note for this table
        sample_list (pd.Series): samples in the table
        seq_list (pd.Series): sequences in the table

    """

    @property
    def _constructor_expanddim(self):
        logging.error("Expand dimension constructor is not defined", error_type=NotImplemented)
        return None

    def get_info(self):
        seq, sample = self.shape
        r = "Table with {} sequences in {} samples (unit: {})".format(seq, sample, self.unit)
        if self.note:
            r += '\n' + self.note
        return r

    def __init__(self, data, sample_list=None, seq_list=None, unit='count', note=None, use_sparse=True, **kwargs):

        if use_sparse:
            dtype = pd.SparseDtype(
                'int' if (unit is None or unit.lower() in ['count', 'counts', 'read', 'reads']) else 'float',
                fill_value=0
            )
        else:
            dtype = int if unit.lower() in ['count', 'counts', 'read', 'reads'] else float

        if isinstance(data, pd.DataFrame):
            super().__init__(data.values, index=data.index, columns=data.columns, dtype=dtype, **kwargs)
        elif isinstance(data, (np.ndarray, list)):
            if (sample_list is None) or (seq_list is None):
                logging.error("Please provide sample_list and seq_list if data is np.ndarray", error_type=ValueError)
            super().__init__(data, index=seq_list, columns=sample_list, dtype=dtype, **kwargs)
        else:
            logging.error("data should be pd.DataFrame or 2-D np.ndarray", error_type=TypeError)
        self.unit = unit
        self.note = note
        self.is_sparse = use_sparse

    def describe(self, percentiles=None, include=None, exclude=None):
        if self.is_sparse:
            return self.sparse.to_dense().describe(percentiles=percentiles, include=include, exclude=exclude)
        else:
            return super().describe(percentiles=percentiles, include=include, exclude=exclude)

    @property
    def samples(self):
        return self.columns.to_series()

    @property
    def seqs(self):
        return self.index.to_series()


class SeqTable(object):
    """Contingency table for sequences in each samples


    Attributes:
        table (Tables): a collection of table for analysis. Including at least `original` created during initialization


    todo: fill up docstring
    Methods:

    Attributes:
        table_original (pd.DataFrame): default original table when initialize a `SeqTable` instance


    Plugins:
    """

    # def __repr__(self):
    #     # todo: update to include key information for the seq table
    #     #      include list of table, number of samples, number of sequences
    #     pass

    def __init__(self, data, data_unit=None, sample_list=None, seq_list=None, data_note=None, use_sparse=True,
                 seq_metadata=None, sample_metadata=None,
                 grouper=None, x_values=None, x_unit=None, note=None, dataset_metadata=None, silent=True):
        """Initialize a `SeqTable` object

        Args:
            data (pd.DataFrame or np.ndarray): 2-D data with indices as sequences and columns as samples. If data is
                pd.DataFrame, values in index and column will be used as sequences and samples; if data is a 2-D
                np.ndarray, `sample_list` and `seq_list` are needed with same length and order as data
            data_unit (str): optional.

        todo:
          - use hidden variable and seq/sample list for masking count table, amount table, dna amount

        Find all valid sequences that occur at least once in any 'input' sample and once in any 'reacted' sample
        """

        # initialize metadata
        from datetime import datetime
        self.metadata = AttrScope(created_time=datetime.now(), note=note)
        # add metadata
        if dataset_metadata is not None:
            self.metadata.add(dataset_metadata)
        if sample_metadata is not None:
            self.metadata_samples = AttrScope(sample_metadata)
        if seq_metadata is not None:
            self.metadata_seqs = AttrScope(seq_metadata)
        logging.info('SeqTable created')

        # add original table
        self.table = AttrScope(
            original=Table(data=data, sample_list=sample_list, seq_list=seq_list, unit=data_unit, note=data_note,
                           use_sparse=use_sparse)
        )

        # add x values
        if x_values is None:
            self.x_values = None
            self.x_unit = None
        elif isinstance(x_values, dict):
            self.x_values = pd.Series(x_values)
            self.x_unit = x_unit
        elif isinstance(x_values, (list, np.ndarray)):
            self.x_values = pd.Series(x_values, index=self.table.original.samples)
            self.x_unit = x_unit
        else:
            logging.error('Unknown type for x_values', error_type=TypeError)

        # import grouper
        self.grouper = AttrScope()
        if grouper is not None:
            from .grouper import Grouper
            if isinstance(grouper, Grouper):
                self.grouper.add(default=grouper)
            else:
                self.grouper.add({name: Grouper(group=value, target=self.table.original)
                                  for name, value in grouper.items()})

        # from .visualizer import seq_occurrence_plot, rep_variability_plot
        # from ..utility.func_tools import FuncToMethod
        # self.visualizer = FuncToMethod(obj=self,
        #                                functions=[
        #                                    seq_occurrence_plot,
        #                                    rep_variability_plot
        #                                ])

    @property
    def sample_list(self):
        return self.table.original.sample_list

    @sample_list.setter
    def sample_list(self, sample_list):
        logging.error("sample_list is inferred from original table and should not be changed",
                      error_type=PermissionError)

    @property
    def seq_list(self):
        return self.table.original.seq_list

    @seq_list.setter
    def seq_list(self, seq_list):
        logging.error("seq_list is inferred from original table and should not be changed",
                      error_type=PermissionError)

    def add_spike_in(self, base_table, spike_in_seq, spike_in_amount, radius=2, unit=None, dist_type='edit'):
        """Accessor to add SpikeInNormalizer"""
        from .transform import SpikeInNormalizer
        if isinstance(base_table, str):
            base_table = getattr(self.table, base_table)
        setattr(self, 'spike_in',
                SpikeInNormalizer(base_table=base_table, spike_in_seq=spike_in_seq, spike_in_amount=spike_in_amount,
                                  radius=radius, unit=unit, dist_type=dist_type))

    def add_sample_total_amounts(self, total_amounts, full_table, unit=None):
        """Add TotalAmountNormalizer to quantify sequences with their total amount in each sample
          as `sample_total_amounts`

        Args:
            total_amounts (dict or pd.Series): total amount for each sample
            full_table (str or pd.DataFrame): corresponding table total amount measured with.
              Get attributes of the instance if it is str
            unit (str): unit for the amount measured
        """
        from .transform import TotalAmountNormalizer
        if isinstance(full_table, str):
            full_table = getattr(self.table, full_table)

        setattr(self, 'sample_total_amounts',
                TotalAmountNormalizer(full_table=full_table,
                                      total_amounts=total_amounts,
                                      unit=unit))

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
            try:
                info[f"dna amount (from total dna{'' if self.dna_amount is None else ', ' + self.dna_amount.unit})"] = \
                    self.dna_amount.dna_amount[sample_name]
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

    def to_json(self):
        """More generalized JSON file
        TODO: add to_json and from_json
        """
        pass

    def from_json(self):
        pass

    def to_pickle(self, path):
        import pickle
        with open(path, 'wb') as handle:
            pickle.dump(self, handle, protocol=-1)

    @staticmethod
    def from_pickle(path):
        import pickle
        with open(path, 'rb') as handle:
            return pickle.load(handle)

    @staticmethod
    @_table_doc.compose(f"""Create a ``SeqTable`` instance from a folder of count files

    Args:
        count_files (str): root directory to search for count files
        file_list (list of str): optional, only includes the count files with names in the file_list
        pattern_filter (str): optional, filter file names based on this pattern, wildcards ``*/?`` are allowed
        black_list (list of str): optional, file names included in black_list will be excluded
        name_template (str): naming convention to extract metadata. Use ``[...]`` to include the region of sample_name,
            use ``{{domain_name[, int/float]}}`` to indicate region of domain to extract as metadata, including
            ``int`` or ``float`` will convert the domain value to int/float in applicable, otherwise, string
        sort_by (str): sort the order of samples based on given domain
        dry_run (bool): only return the parsed count file names and metadata without actual reading in data
    <<x_values, x_unit, input_sample_name, sample_metadata, note>>

    {_name_template_example}

    """)
    def from_count_files(**kwargs):
        from .count_file import load_Seqtable_from_count_files
        return load_Seqtable_from_count_files(**kwargs)

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

    #     #




#     #
#     # TODO: consider add accessor to fitting
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
#     #     self.logger.info('BatchFitting fitter added')
#     #
#     # @classmethod
#     # def load_count_files(cls, folder_path, x_values, file_pattern=None, name_template=None, file_list=None, black_list=None,
#     #                      x_unit=None, dna_unit=None,
#     #                      spike_in_seq=None, spike_in_amount=None, spike_in_dia=None, ):
#     #     """To accommodate for exploratery needs, this will be a wrapper for `count_file.CountFileSet"""
#     #     note = f'Loaded from {folder_path}'
#     #



def slice_table(table, axis, keys=None, filter_fn=None, remove_zero=False):
    """Utility function to slice pd.DataFrame table with a list of key values or filter functions along given axis
    Optional to remove all zero entries
    Args:
        table (pd.DataFrame): name table to slice
        keys (list-like): list of keys to preserve
        axis (0 or 1)
        remove_zero (bool): if remove all zero items in the other axis
    TODO: refactor slice function, collect other similar functions and move here
    """

    if keys is None:
        logging.error('`keys` or `mask` must be provided')
        raise ValueError('`keys` or `mask` must be provided')

    if axis == 0:
        if keys is not None:
            sub_table = table.loc[keys]
        if remove_zero:
            return sub_table.loc[:, (sub_table != 0).any(axis=0)]
        else:
            return sub_table
    else:
        if keys is not None:
            sub_table = table[keys]
        if remove_zero:
            return sub_table.loc[(sub_table != 0).any(axis=1)]
        else:
            return sub_table
