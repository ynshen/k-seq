"""Submodule of `SeqData`, a rich functions class of table for sequencing manipulation This module contains methods
for data pre-processing from count files to ``CountFile`` for estimator
For absolute quantification, it accepts absolute amount (e.g. measured by qPCR) or reacted fraction
TODO:
  - write output function for each class as JSON file
"""
import numpy as np
import pandas as pd
from ..utility.func_tools import AttrScope, FuncToMethod
from ..utility.file_tools import _name_template_example
from ..utility.log import logging, Logger
from .transform import _spike_in_doc, _total_dna_doc
from doc_helper import DocHelper


_doc = DocHelper(
    x_values=('list-like, or dict', 'optional. value for controlled variables. If list-like, should have same length '
                                    'and order as samples; if dict, should have sample names as key'),
    x_unit=('str', "optional. Unit for controlled variable. e.g. 'uM'"),
    input_sample_name=('list of str', 'optional. Indicate input samples (unreacted)'),
    sample_metadata=('dict of objects', 'optional. Extra sample metadata'),
    seq_metadata=('dict of objects', 'optional. Extra seq metadata'),
    dataset_metadata=('dict of objects', 'optional. Extra dataset metadata'),
    note=('str', 'Note for dataset/table'),
    data=('pd.DataFrame or np.ndarray', '2-D data with indices as sequences and columns as samples. '
                                        'If data is pd.DataFrame, values in index and column will be used as '
                                        'sequences and samples; if data is a 2-D np.ndarray, `sample_list` and '
                                        '`seq_list` are needed with same length and order as data'),
    sample_list=('list-like', 'list of samples in the sample, should match the columns in the table data'),
    seq_list=('list-like', 'list of seqs in the sample, should match the rows in the table data'),
    data_unit=('str', 'The unit of table values, e.g. counts, ng, M. Default counts.'),
    use_sparse=('bool', 'If store the table value as sparse matrix'),
    data_note=('str', 'Note for data table'),
    remove_empty=('bool', 'If remove the empty column/rows after filtering')

)


class SeqTable(pd.DataFrame):
    """Enhanced ``pd.DataFrame`` with added property and functions for SeqData

    Additional Attributes:
        unit (str): unit of entries in this table
        note (str): note for this table
        samples (pd.Series): samples in the table
        seqs (pd.Series): sequences in the table

    Additional Methods:
      about: print a summary of the table
      TODO: add Table-wise visualization here

    """

    @_doc.compose("""Initialize SeqTable instance
    Args:
    <<data, sample_list, seq_list, unit, note, use_sparse>>
    """)
    def __init__(self, data, sample_list=None, seq_list=None, unit='count', note=None, use_sparse=True, **kwargs):

        if use_sparse:
            dtype = pd.SparseDtype(
                'int' if (unit is not None and unit.lower() in ['count', 'counts', 'read', 'reads']) else 'float',
                fill_value=0
            )
        else:
            dtype = int if unit.lower() in ['count', 'counts', 'read', 'reads'] else float

        if isinstance(data, (pd.DataFrame, SeqTable)):
            super().__init__(data.values, index=data.index, columns=data.columns, dtype=dtype, **kwargs)
        elif isinstance(data, (np.ndarray, list)):
            if (sample_list is None) or (seq_list is None):
                logging.error("Please provide sample_list and seq_list if data is np.ndarray", error_type=ValueError)
            super().__init__(data, index=seq_list, columns=sample_list, dtype=dtype, **kwargs)
        else:
            super().__init__(data, **kwargs)
        self.unit = unit
        self.note = note
        self.is_sparse = use_sparse
        self.vis = FuncToMethod([seq_overview, sample_overview],
                                obj=self)

    @property
    def _constructor_expanddim(self):
        """abstract method needed to implemented, not used"""
        logging.error("Expand dimension constructor is not defined", error_type=NotImplemented)
        return None

    def about(self):
        """Quick view of SeqTable"""

        seq, sample = self.shape
        r = "SeqTable with {} sequences in {} samples \ndensity: {:.3f}, unit: {}, memory: {:.3f} KB".format(
            seq, sample, self.density, self.unit, self.memory_usage().sum() / 1024)
        if self.note:
            r += '\n' + self.note
        print(r)

    def describe(self, percentiles=None, include=None, exclude=None):
        """return major stable statistics"""
        if self.is_sparse:
            return self.sparse.to_dense().describe(percentiles=percentiles, include=include, exclude=exclude)
        else:
            return super().describe(percentiles=percentiles, include=include, exclude=exclude)

    @property
    def samples(self):
        return pd.Series(self.columns)

    @property
    def seqs(self):
        return pd.Series(self.index)

    @property
    def density(self):
        if not hasattr(self, '_density'):
            self._density = ((self == 0) | np.isnan(self)).sum().sum() / self.shape[0] / self.shape[1]
        return self._density

    @_doc.compose("""Filter table along with one axis

    Args:
        filter (callable): a callable to apply on row/columns and returns a bool value
        axis (0 or 1): the axis to filter. 0: row, seqs; 1: column, sample
        <<remove_empty>>
        inplace (bool): if change the table inplace. If False, return a new table
    """)
    def filter_axis(self, filter, axis=0, remove_empty=False, inplace=False):

        allowed_axis = {
            'sample': 1,
            'observation': 1,
            1: 1,
            'seq': 0,
            'sequences': 0,
            'seqs': 0,
            0: 0
        }
        if isinstance(axis, str):
            axis = axis.lower()
        if axis not in allowed_axis.keys():
            logging.error("Unknown axis, please use 'sample'/1 or 'sequence'/1", error_type=ValueError)
        else:
            axis = allowed_axis[axis]

        if inplace:
            sliced = slice_table(self, axis=axis, keys=filter, remove_empty=remove_empty)
            self.reindex(index=sliced.index, columns=sliced.columns, copy=False)
        return slice_table(self, axis=axis, keys=filter, remove_empty=remove_empty)


@_doc.compose("""Slice pd.DataFrame table with a list of key values or filter functions returning True/False along 
given axis. Optional to remove all zero entries
Args:
    table (pd.DataFrame): table to slice
    keys (list-like or callable): list of keys to preserve. If is callable, apply to row/column of table and returns
        bool of preserve (True) or discard (False)
    axis (0 or 1): which axis to filter
    <<remove_empty>>
""")
def slice_table(table, axis, keys, remove_empty=False):

    if callable(keys):
        keys = table.apply(keys, axis=1 - axis)
        keys = keys[keys].index.values

    if axis == 0:
        sub_table = table.loc[keys]
        if remove_empty:
            return sub_table.loc[:, (sub_table != 0).any(axis=0)]
        else:
            return sub_table
    else:
        sub_table = table[keys]
        if remove_empty:
            return sub_table.loc[(sub_table != 0).any(axis=1)]
        else:
            return sub_table


def seq_overview(table, axis=0):
    """Summarize sample for a given table, with info of seq length, sample detected, mean, sd
    Returns:
        A `pd.DataFrame` show the summary for sequences
    """
    if axis == 1:
        table = table.transpose()

    return pd.DataFrame.from_dict(
        {'length': table.index.to_series().apply(len),
         'samples detected': (table > 0).sum(axis=1),
         'mean': table.mean(axis=1),
         'sd': table.std(axis=1)},
        orient='columns'
    )


def sample_overview(table, axis=1):
    """Summarize sequences for a given table, with info of unique seqs, total amount

    Returns:
        A `pd.DataFrame` show the summary for sequences
    """
    if axis == 0:
        table = table.transpose()

    if isinstance(table, SeqTable):
        col_name = f'total amount ({table.unit})'
    else:
        col_name = 'total amount'

    return pd.DataFrame.from_dict(
        {'unique seqs': (table > 0).sum(axis=0),
         col_name: table.sum(axis=0)},
        orient='columns'
    )


@_doc.compose("""Data instance to store k-seq result

Attributes:
    table (AttrScope): accessor for tables stored. Including ``original`` created during initialization.
        tables stored should be ``pd.DataFrame`` or ``SeqTable``
<<x_values, x_unit>>
    seqs
    samples
    metadata (AttrScope): accessor for metadata for the dataset, includes
        sample (AttrScope): collection of metadata for samples if applicable
        seq (AttrScope): collection of metadata for seqs if applicable
        created_time (timestamp): datetime of the instance is created
        note (str): note for the dataset
        other dataset metadata objects could be added
    
    vis (AttrScope): accessor to some pre-built visualizations

Plugins:
    grouper (GrouperCollection): collection of ``Grouper`` to slice subtables
    spike_in (SpikeInNormalizer): optional. Accessor to the normalizer using spike-in
    sample_total (TotalAmountNormalizer): optional. Accessor to the normalizer using total sample amount of seqs  

Methods:
    TODO: add methods



""")
class SeqData(object):

    def __repr__(self):
        repr = f"SeqData ({super().__repr__()})\n"
        repr += f"Includes {len(self.seqs)} seqs, {len(self.samples)} samples\nNote: {self.metadata.note}\n"
        repr += 'Tables:\n'
        for name, table in self.table.__dict__.items():
            repr += f"\t{name} {table.shape}\n"
        return repr

    @_doc.compose("""Initialize a `SeqData` object

    Args:
    <<data, data_unit, sample_list, seq_list, data_note, use_sparse, seq_metadata, grouper, x_values, x_unit, note, dataset_metadata>>
    grouper (dict of list, or dict of dict of list): optional. dict of list (Type 1) of dict of list (Type 2) to 
        create grouper plugin
    """)
    def __init__(self, data, data_unit=None, sample_list=None, seq_list=None, data_note=None, use_sparse=True,
                 seq_metadata=None, sample_metadata=None,
                 grouper=None, x_values=None, x_unit=None, note=None, dataset_metadata=None):

        # initialize metadata
        from datetime import datetime
        self.metadata = AttrScope(created_time=datetime.now(), note=note)
        # add metadata
        if dataset_metadata is not None:
            self.metadata.add(dataset_metadata)
        if sample_metadata is not None:
            self.metadata.samples = AttrScope(sample_metadata)
        if seq_metadata is not None:
            self.metadata.seqs = AttrScope(seq_metadata)
        logging.info('SeqData created')

        # add original table
        self.table = AttrScope(original=SeqTable(data=data, sample_list=sample_list, seq_list=seq_list,
                                                 unit=data_unit, note=data_note, use_sparse=use_sparse))

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

        if grouper is not None:
            from .grouper import GrouperCollection
            self.grouper = GrouperCollection()
            self.grouper.add(**grouper)

        # from .visualizer import seq_occurrence_plot, rep_variability_plot
        # from ..utility.func_tools import FuncToMethod
        # self.visualizer = FuncToMethod(obj=self,
        #                                functions=[
        #                                    seq_occurrence_plot,
        #                                    rep_variability_plot
        #                                ])

    @property
    def samples(self):
        return self.table.original.samples

    @samples.setter
    def samples(self, samples):
        logging.error("samples is inferred from original table and should not be changed",
                      error_type=PermissionError)

    @property
    def seqs(self):
        return self.table.original.seqs

    @seqs.setter
    def seqs(self, seqs):
        logging.error("seqs is inferred from original table and should not be changed",
                      error_type=PermissionError)

    @_spike_in_doc.compose("""Add SpikeInNormalizer to quantify seq amount using spike-in sequence as accessor 
    ``spike_in`` to the instance
    
    Args:
        <<base_table, spike_in_seq, spike_in_amount, radius, unit, dist_type>>
    """)
    def add_spike_in(self, base_table, spike_in_seq, spike_in_amount, radius=2, unit=None, dist_type='edit'):

        from .transform import SpikeInNormalizer
        if isinstance(base_table, str):
            base_table = getattr(self.table, base_table)
        setattr(self, 'spike_in',
                SpikeInNormalizer(base_table=base_table, spike_in_seq=spike_in_seq, spike_in_amount=spike_in_amount,
                                  radius=radius, unit=unit, dist_type=dist_type))

    @_total_dna_doc.compose("""Add TotalAmountNormalizer to quantify sequences with their total amount in each sample
      as `sample_total`

    Args:
        <<total_amounts, full_table, unit>>
    """)
    def add_sample_total(self, total_amounts, full_table, unit=None):
        from .transform import TotalAmountNormalizer
        if isinstance(full_table, str):
            full_table = getattr(self.table, full_table)

        setattr(self, 'sample_total',
                TotalAmountNormalizer(full_table=full_table,
                                      total_amounts=total_amounts,
                                      unit=unit))

    def to_json(self):
        """More generalized JSON file
        TODO: add to_json and from_json
        """
        pass

    def from_json(self):
        """TODO: add json"""
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
    @_doc.compose(f"""Create a ``SeqData`` instance from a folder of count files

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

    def sample_info(self):
        """Summarize sample info for a SeqData, with info of
        Returns:
            A `pd.DataFrame` show the summary for samples
        """
        info = pd.DataFrame(index=self.samples)
        if hasattr(self, 'grouper') and hasattr(self.grouper, 'input'):
            def get_sample_type(sample):
                if sample in self.grouper.input.group:
                    return 'input'
                elif sample in self.grouper.reacted.group:
                    return 'reacted'
                else:
                    return np.nan
            info['sample type'] = info.index.to_series().apply(get_sample_type)

        if self.x_values is not None:
            info['x values'] = self.x_values

        info = pd.concat([info, self.table.original.vis.sample_overview()], axis=1)

        if hasattr(self, 'spike_in'):
            info['total amount (spike-in)'] = self.spike_in.norm_factor * self.spike_in.base_table.sum(axis=0)
            info['spike-in fraction'] = self.spike_in.peak.peak_abun(use_relative=False)[0].sum(axis=0) / self.spike_in.base_table.sum(axis=0)
        
        if hasattr(self, 'sample_total'):
            info['total amount (sample total)'] = pd.Series(self.sample_total.total_amounts)

        return info


#     # TODO: consider add accessor to fitting
#     # def add_fitting(self, model, seq_to_fit=None, weights=None, bounds=None,
#     #                 bootstrap_depth=0, bs_return_size=None,
#     #                 resample_pct_res=False, missing_data_as_zero=False, random_init=True, metrics=None):
#     #     """
#     #     Add a `k_seq.estimator.BatchFitting` instance to SeqData for estimator
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



