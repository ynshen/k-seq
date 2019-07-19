"""This module contains methods for data preprocessing from count files to ``CountFile`` for fitting
TODOs:
  - write output function for each class as JSON file
"""


class SeqSample:
    """This class describes experimental samples sequenced in k-seq
    """

    def __init__(self, file_path, x_value, name_pattern=None, load_data=False, silent=False):
        """`SeqSample` instance store the count file data for each sequencing sample in k-seq experiments
        Initialize a `SeqSample` instance by linking to a count file
        Args:
            file_path (`str`): directory to the count file of sample

            x_value (`float` or `str`): x-axis value for the sample (e.g. time point or substrate concentration) in the fitting.
                If `str`, it will extract value from the domain indicated by ``name_pattern``

            name_pattern (`str`): optional. Pattern to automatically extract metadata using
                :func:`~k_seq.utility.extract_metadata` (click to see details). Briefly,

                - Use ``[...]`` to include the region of sample_name (required),

                - Use ``{domain_name[, int/float]}`` to indicate region of domain to extract as metadata,

                including ``[,int/float]`` will convert the domain value to int/float if applicable, otherwise, string.

            load_data (`bool`): if load data from the count file when initializing the object. Recommend to set as False
                for large files. Default False.

            silent (bool): Print progress to std.out if False.

        Example:

                .. code-block:: python

                   sample = SeqSample{
                       file_path = "path/to/count/file/R4B-1250A_S16_counts.txt",
                       x_value = 'concentration',
                       name_pattern = "R4[{exp_rep}-{concentration, float}{seq_rep}]_S{id, int}_counts.txt"
                       load_data = False,
                       silent = False
                    }

            will return a ``SeqSample`` instance ``sample`` that

                .. code-block:: python

                    > sample.name
                    'B-1250A'
                    > sample.metadata
                    {
                        'exp_rep': 'B',
                        'concentration': 1250.0,
                        'seq_rep': 'A',
                        'id': 16
                    }
                    > sample.sequences
                    Error

        Attributes:

            name (`str`): name of the sample, extracted from the file name

            unique_seqs (``int``): number of unique sequences in the sample

            total_counts (``int``): number of total reads in the sample

            sequences (``dict``): a dictionary of counts for each unique sequences
                {seq: counts}

            sample_type (``str``, 'input' or 'reacted'): 'input' if sample is from initial pool;
                'reacted' if sample is reacted pool

            x_value (float): the value of time point or concentration point for the sample

            metadata (dict): a dictionary contains metadata of the sample:

                - file_path (`str`): path to the count file

                - log (`str`): logging of all modification applied on the data

                - Other metadata extracted from the file name
        """

        from k_seq import utility
        from pathlib import Path
        import numpy as np
        import pandas as pd

        self.metadata = {}
        self._logger = utility.Logger()
        file_path = Path(file_path)
        self.metadata['file_path'] = str(file_path)
        if not silent:
            print("Creating sample from {}".format(self.metadata['file_path']))

        if name_pattern:
            metadata = utility.extract_metadata(target=file_path.name, pattern=name_pattern)
            self.name = metadata.pop('name', None)
            self.metadata.update(metadata)
        else:
            self.name = file_path.name

        if 'input' in self.name or 'Input' in self.name:
            self.sample_type = 'input'
            self.x_value = np.nan
        else:
            self.sample_type = 'reacted'
            if isinstance(x_value, str):
                self.x_value = self.metadata[x_value]
            else:
                self.x_value = x_value

        self._logger.add_log('SeqSample instance created')
        if load_data:
            self.load_data(silent=silent)

        # Import visualizers
        from ..utility import FunctionWrapper
        from .visualizer import length_dist_plot_single, sample_count_cut_off_plot_single
        self.visualizer = FunctionWrapper(data=self, functions=[length_dist_plot_single,
                                                                sample_count_cut_off_plot_single])

    def load_data(self, silent=False):
        """Load data from file with path in ``self.matadata['file_path']``"""

        from k_seq.data import io

        if not silent:
            print("Load count data from file {}".format(self.metadata['file_path']))
        self.unique_seqs, self.total_counts, self._sequences = io.read_count_file(self.metadata['file_path'])
        self._logger.add_log('Data imported from file')

    def survey_spike_in_peak(self, spike_in_seq, max_dist_to_survey=10, silent=False, inplace=True):
        """Survey spike-in counts in the sample.
        Calculate the total counts of sequences that is *i* (*i* from 0 to `max_dist_to_survey`) edit distance from
            exact spike-in sequences (external standard for quantification) in the sample.
            Add attribute ``spike_in`` to the instance if `inplace` is True.

        Args:
            spike_in_seq (`str`): the exact sequence of spike-in

            max_dist_to_survey (`int`): the maximum distance to survey

            silent (`bool`): don't print progress if True

            inplace (`bool`): Add attribute ``spike_in`` if True, else return a dict object
                {
                    'spike_in_seq': str, spike in sequence,
                    'spike_in_counts': list of `int`, number of sequences that is *i* edit distance away
                }

        Attributes:

            spike_in (`dict`):

                - spike_in_counts (`list` of `int`): length `max_dist_to_survey + 1` list, number of total counts with
                    different edit distance to exact spike-in sequence

                - spike_in (`str`): exact spike in sequence
        """

        import Levenshtein
        import numpy as np
        import pandas as pd

        results = dict()
        if not silent:
            print("Survey spike-in counts for sample {}...".format(self.name))
        results['spike_in_seq'] = spike_in_seq
        edit_dist = lambda seq: Levenshtein.distance(spike_in_seq, seq)
        self._sequences['dist_to_spike_in'] = self._sequences.index.map(edit_dist)
        dists = np.linspace(0, max_dist_to_survey, max_dist_to_survey + 1, dtype=np.int)
        results['spike_in_peak'] = pd.DataFrame(index=dists)
        results['spike_in_peak']['unique_seq'] = [np.sum(self._sequences['dist_to_spike_in'] == dist) for dist in dists]
        results['spike_in_peak']['total_counts'] = [
            np.sum(self._sequences[self._sequences['dist_to_spike_in'] == dist]['counts'])
            for dist in dists
        ]

        if inplace:
            self.spike_in = results
            self._logger.add_log('Spike-in sequences surveyed on with maximal distance {}'.format(max_dist_to_survey))
        else:
            return results

    def get_quant_factor(self, from_spike_in_amount=None, spike_in_seq=None, max_dist=0, from_total_amount=None,
                         silent=False):
        """Calculate quantification factor for the sample, either from spike in or total amount
        If `from_spike_in_amount` is not `None`, will priory use spike in to quantify the amount of each sequence, and
            attributes ``quant_factor`` and ``quant_factor_max_dist`` will be add to the instance,
        If `from_spike_in_amount` is `None`, we expect `from_total_amount` to be not `None`

        Args:

            from_spike_in_amount (`float`): Optional. Amount of actual spike-in added in this sample

            max_dist (`int`): maximum edit distance to count a sequence as spike-in, `from_spike_in_amount` has to be not `None`.
                Default 0.

            spike_in_seq (`str`): Optional. Exact spike in sequence, pass to `survey_spike_in` if not performed yet

            from_total_amount (`float`): Optional. Total amount of DNA measured for ths sample

            silent (`bool`): don't print process info if False

        Attributes:

            quant_factor (`float`): defined as
                :math:`\\frac{\\text{spike-in amount}}{\\text{total counts}\\times\\text{spike-in counts[: max_dist + 1]}}`
                if `from_spike_in_amount` is not None
                effectively the total DNA amount in the sequencing pool

            quant_factor_max_dist (`int`): maximum edit distance for a sequence to be spike-in

        """
        import numpy as np

        if not silent:
            print("Calculate quant-factor for sample {}...".format(self.name))

        if from_spike_in_amount:
            if not hasattr(self, 'spike_in'):
                if spike_in_seq is not None:
                    self.survey_spike_in_peak(spike_in_seq, max_dist_to_survey=max_dist, silent=silent, inplace=True)
                else:
                    raise Exception('Please provide spike_in_seq')
            self.quant_factor = float(from_spike_in_amount) * self.total_counts / np.sum(self.spike_in['spike_in_peak']['total_counts'][:max_dist + 1])
            self.spike_in['quant_factor_max_dist'] = max_dist
            self.spike_in['spike_in_amount'] = from_spike_in_amount
            self._logger.add_log(
                'Quantification factor estimated from spike-in seq within distance {}'.format(max_dist)
            )
        elif from_total_amount:
            self.quant_factor = from_total_amount
            self._logger.add_log('Quantification factor estimated by total DNA')
        else:
            raise Exception('Please quantify through either spike-in or total DNA amount')

    def sequences(self, with_spike_in=True, filter=None):
        if with_spike_in:
            seq = self._sequences
        else:
            if hasattr(self, 'spike_in'):
                seq = self._sequences[self._sequences['dist_to_spike_in'] > self.spike_in['quant_factor_max_dist']]
            else:
                seq = Exception('Please calculate quantification factor form spike in first')
        if filter is None:
            return seq
        else:
            if isinstance(filter, list):
                return seq[filter]
            elif callable(filter):
                return seq[seq.apply(func=filter, axis=1)]

    @property
    def log(self):
        return self._logger.log


class SeqSampleSet:
    """Object to load and store a set of samples
    """

    def __init__(self, file_root, x_values, count_file_pattern=None, name_pattern=None,
                 sort_by=None, file_list=None, black_list=None, load_data=False, silent=True, note=None):
        """Initialize by linking count files under a root folder into :func:`~SeqSample` objects

        Args:
            file_root (`str`): directory to the root folder for count files

            x_values (`str` or list of `float`): Time points or concentration points values for samples. If `str`, the
              function will use it as the domain name to extract value from file name; if a list
              of `float`, it should have same length and order as sample file under `file_root`
              (Use :func:`~k_seq.utility.get_file_list' to examine the files automatically extracted)

            count_file_pattern (`str`): optional, name pattern to identify count files. Only file name contains
              the pattern will be included

            name_pattern (`str`): optional. Pattern to extract metadata, see :func:`~k_seq.utility.extract_metadata'
              for details

            sort_by (`str` or `callable`): optional. If `str`, it should be the domain name to order the sample in the
              ascending order. If `callable`, the input is `SeqSample` instance and the sample will ordered by the
              return value

            file_list (list of `str`): optional, only import the file in sample_list if not `None`

            black_list (list of `str`): optional, name of sample files that are excluded

            load_sample (`bool`): if read and load count file during initialization. Recommand to be False for large
              count files, default False.

            silent (boolean): don't print process if True

        """
        from pathlib import Path
        from k_seq import utility
        import numpy as np
        import pandas as pd

        self._logger = utility.Logger()
        self._logger.add_log('Dataset created from {}'.format(file_root))
        self.metadata = {
            'file_root': file_root
        }
        if note:
            self.metadata['note'] = note
        else:
            self.metadata['note'] = None
        if black_list:
            self.metadata['black_list'] = black_list
        else:
            self.metadata['black_list'] = None
            black_list = []
        if file_list is None:
            file_list = [file.name for file in Path(file_root).glob('*{}*'.format(count_file_pattern))]
            print("NOTICE: no sample list is given, samples are collected from root folder:\n\t" + '\n\t'.join(file_list))
        self.metadata['file_list'] = file_list
        if isinstance(x_values, list):
            if len(x_values) != len(file_list):
                raise Exception("Errors: sample_list is given as a list, but the length does not match with "
                                "sample files")
        else:
            x_values = [x_values for _ in file_list]
        self.metadata['x_values'] = x_values
        self.sample_set = []
        for file_name, x_value in zip(file_list, x_values):
            if file_name not in black_list:
                self.sample_set.append(SeqSample(file_path=str(Path.joinpath(Path(file_root), file_name)),
                                                 x_value=x_value,
                                                 name_pattern=name_pattern,
                                                 load_data=load_data,
                                                 silent=silent))

        if load_data:
            self._logger.add_log('Data loaded as dataset created')

        if sort_by:
            if isinstance(sort_by, str):
                sort_fn = lambda single_file: single_file.metadata[sort_by]
            elif callable(sort_by):
                sort_fn = sort_by
            self.sample_set = sorted(self.sample_set, key=sort_fn)

        print("Samples imported from {}".format(file_root))

        from .visualizer import count_file_info_table, count_file_info_plot, spike_in_peak_plot, rep_spike_in_plot, length_dist_plot_all, sample_count_cut_off_plot_all
        from ..utility import FunctionWrapper
        self.visualizer = FunctionWrapper(data=self,
                                          functions=[
                                              count_file_info_table,
                                              count_file_info_plot,
                                              spike_in_peak_plot,
                                              rep_spike_in_plot,
                                              length_dist_plot_all,
                                              sample_count_cut_off_plot_all
                                          ])

    def load_data(self, silent=True):
        """Load data after creating the object, suitable for large files"""
        for sample in self.sample_set:
            sample.load_data(silent=silent)
        self._logger.add_log('Data loaded at from {}'.format(self.metadata['file_root']))

    def get_quant_factors(self, from_spike_in_amounts=None, spike_in_seq=None, max_dist=2,
                          max_dist_to_survey=None, from_total_amounts=None, quant_factors=None,
                          survey_only=False, silent=True):
        """Calculate quantification factors for each sample in `SeqSampleSet`.
        This method will first survey the spike-in sequence, then calculate the quantification factor for each sample if
        applies.

        Args:
            from_spike_in_amounts (list/dict of `float`): optional. Use if using spike in for quantification.
              A list/dict of absolute amount of spike-in sequence in each sample
              If `dict`, the key must match the names of samples; if `list`, the length and order much match the `sample_names`

            spike_in_seq (`str`): optional. Spike in sequence if using spike in for quantification.

            max_dist (`int`): optional, maximum edit distance to exact spike-in sequence to be counted as spike-in.
              Default: 2

            max_dist_to_survey (`int`): optional, survey the spike-in counts with maximum edit distance. If `None`,
              `max_dist` will be used

            from_total_amounts (list/dict of 'float'): optional. Use if using total DNA amount for quantification. Same
              as `from_spike_in_amount`, a list/dict of total amount of DNA in each sample. If `dict`, the key must
              match the names of samples; if `list`, the length and order much match the `sample_names`

            quant_factors (list/dict of `float`): optional, accept a list/dict of manually curated quant_factor. If `dict`,
              the key must match the names of samples; if `list`, the length and order much match the `sample_names`

            survey_only (`bool`): optional. If True, only survey the spike-in counts within maximum edit distance

            silent (`bool`): don't print process if True

        """

        if quant_factors:
            # curated quant factors has highest priority
            if len(quant_factors) == len(self.sample_set):
                if isinstance(quant_factors, dict):
                    for sample in self.sample_set:
                        if sample.name in quant_factors.keys():
                            sample.quant_factor = quant_factors[sample.name]
                elif isinstance(quant_factors, list):
                    for sample, quant_factor in zip(self.sample_set, quant_factors):
                        sample.quant_factor = quant_factor
            else:
                raise Exception('Error: input quant_factor has different number as samples')
        elif from_spike_in_amounts:
            if spike_in_seq:
                if max_dist_to_survey is None:
                    max_dist_to_survey = max_dist
                    if not silent:
                        print('Use max_dist ({}) as max_dist_to_survey as it is not provided'.fromat(max_dist))
                for sample in self.sample_set:
                    sample.survey_spike_in(spike_in_seq=spike_in_seq,
                                           max_dist_to_survey=max_dist_to_survey,
                                           silent=silent,
                                           inplace=True)
            elif not hasattr(self.sample_set[0], 'spike_in'):
                raise Exception('Please indicate spike in sequence')
            if not survey_only:
                if isinstance(from_spike_in_amounts, dict):
                    for sample in self.sample_set:
                        if sample.name in from_spike_in_amounts.keys():
                            sample.get_quant_factor(from_spike_in_amount=from_spike_in_amounts[sample.name],
                                                    max_dist=max_dist, silent=silent)
                elif isinstance(from_spike_in_amounts, list):
                    for sample, spike_in_amount in zip(self.sample_set, from_spike_in_amounts):
                        sample.get_quant_factor(from_spike_in_amount=spike_in_amount,
                                                max_dist=max_dist, silent=silent)
                else:
                    raise Exception('from_spike_in_amounts should be either list or dict')
        elif from_total_amounts:
            if isinstance(from_total_amounts, dict):
                for sample in self.sample_set:
                    if sample.name in from_total_amounts.keys():
                        sample.get_quant_factor(from_total_amount=from_total_amounts[sample.name], silent=silent)
            elif isinstance(from_total_amounts, list):
                for sample, total_amount in zip(self.sample_set, from_total_amounts):
                    sample.get_quant_factor(from_total_amount=total_amount, silent=silent)
            else:
                raise Exception('from_total_amounts should be either list or dict')

    def get_samples(self, sample_id, with_spike_in=False, return_SeqSample=False):
        """Return (count) info of sample(s)

        Args:
            sample_id (`str`, list of `str`, or `callable`): indicate what sample to select
            return_SeqSample (`bool`): return a list of `SeqSample` if True; return a `pd.DataFrame` of counts if False

        Returns: list of `SeqSample` or `pd.DataFrame`

        """
        import pandas as pd

        if isinstance(sample_id, str):
            sample_to_return = [sample for sample in self.sample_set if sample.name == sample_id]
        elif isinstance(sample_id, list):
            sample_to_return = [sample for sample in self.sample_set if sample.name in sample_id]
        elif callable(sample_id):
            sample_to_return = [sample for sample in self.sample_set if sample_id(sample)]
        else:
            raise Exception('Error: please pass sample id, a list of sample id, or a callable on SeqSample')

        if return_SeqSample:
            return sample_to_return
        else:
            seq_list = set()
            for sample in sample_to_return:
                seq_list.update(list(sample.sequences(with_spike_in).index))
            return_df = pd.DataFrame(index=seq_list)
            for sample in sample_to_return:
                return_df[sample.name] = sample.sequences(with_spike_in)
            return return_df

    def filter_sample(self, sample_to_keep, inplace=True):
        """filter samples in sample set

        Args:
            sample_to_keep (list of `str`): name of samples to keep
            inplace (`bool`): return a new SequencingSampleSet if False

        Returns: None if inplace is True, `SeqSampleSet` if inplace is False

        """

        if inplace:
            self.sample_set = [sample for sample in self.sample_set if sample.name in sample_to_keep]
        else:
            import copy
            new_set = copy.deepcopy(self)
            new_set.sample_set = [sample for sample in new_set.sample_set if sample.name in sample_to_keep]
            return new_set

    @property
    def sample_names(self):
        return [sample.name for sample in self.sample_set]

    @property
    def sample_overview(self):
        from . import visualizer
        return visualizer.count_file_info_table(self, return_table=True)

    def to_SeqTable(self, remove_spike_in=True, note=None):
        """Convert to a `SeqTable` object

        Args:
            remove_spike_in (`bool`): remove spike in if True. Default True.
            note (`str`): optional. Note about the sample set

        Returns: `SeqTable` instance
        """
        if note is None:
            note = None
        return SeqTable(self, remove_spike_in=remove_spike_in, note=note)


class SeqTable:
    """This class contains the dataset of valid sequences extracted and aligned from a list of ``SeqSampleSet``
    """

    def __init__(self, sample_set, remove_spike_in=True, note=None):
        """Initialize from a ``SeqSampleSet`` instance
        Find all valid sequences that occur at least once in any 'input' sample and once in any 'reacted' sample

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

            count_table (``pandas.DataFrame``): valid sequences and their original counts in valid samples
        """
        from k_seq.utility import Logger

        self._logger = Logger()
        self._logger.add_log('Dataset created')
        self.metadata = {}
        if note:
            self.metadata['note'] = note

        # find valid sequence set
        self.metadata['remove_spike_in'] = remove_spike_in
        input_set = sample_set.get_sample(sample_id=lambda sample: sample.sample_type == 'input',
                                          with_spike_in=not(remove_spike_in))
        reacted_set = sample_set.get_sample(sample_id=lambda sample: sample.sample_type == 'reacted',
                                            with_spike_in=not(remove_spike_in))
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
                'valid_seqs_num': len(set(sequences.index) & valid_set),
                'valid_seqs_counts': np.sum(sequences[list(set(sequences.index) & valid_set)]['counts'])
            }
            self.sample_info[sample.name].update(sample_info_dict)

        from .visualizer import seq_occurrence_plot, rep_variability_plot
        from ..utility import FunctionWrapper
        self.visualizer = FunctionWrapper(data=self,
                                          functions=[
                                              seq_occurrence_plot,
                                              rep_variability_plot
                                          ])

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
            self._logger.add_log('reacted_frac_tabled added using {} as input average'.format(input_average))
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

    def filter_seq(self, seq_to_keep=None, filter_fn=None, update_all=True, inplace=True):
        """Filter sequence in dataset
        In current version, only ``count_table`` and ``reacted_frac_table`` will change if applicable
        Other meta info will keep the same

        Args:
            seq_to_keep (list of str): sequences to keep
            inplace: change in place if True

        Returns: Return a new SequenceSet object with only ``seq_to_keep`` if inplace is False

        """
        if inplace:
            table_to_use = self
        else:
            import copy
            table_to_use = copy.deepcopy(self)

        if update_all:
            table_to_use.count_table_input = table_to_use.count_table_input.loc[seq_to_keep]
            table_to_use.count_table_reacted = table_to_use.count_table_reacted.loc[seq_to_keep]

        if hasattr(table_to_use, 'reacted_frac_table'):
            table_to_use.reacted_frac_table = table_to_use.reacted_frac_table.loc[seq_to_keep]
        if not inplace:
            return table_to_use

    def add_fitting(self, model, seq_to_fit=None, weights=None, bounds=None,
                    bootstrap_depth=0, bs_return_size=None,
                    resample_pct_res=False, missing_data_as_zero=False, random_init=True, metrics=None):
        """
        Add a `k_seq.fitting.BatchFitting` instance to SeqTable for fitting
        Args:
            model:
            seq_to_fit:
            weights:
            bounds:
            bootstrap_depth:
            bs_return_size:
            resample_pct_res:
            missing_data_as_zero:
            random_init:
            metrics:

        """
        from ..fitting.fitting import BatchFitting
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
            metrics=metrics)
        self._logger.add_log('BatchFitting fitter added')


class SeqFilter:
    # Todo: add some filters for sequences

    def __init__(self):
        pass
