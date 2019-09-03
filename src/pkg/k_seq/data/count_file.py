
class SeqSample(object):

    """This class stores and handles sequencing reads data from experimental samples in k-seq

    Attributes:

            name (`str`): sample name, extracted from the file name

            unique_seqs (`int`): number of unique sequences in the sample

            total_counts (`int`): number of total reads in the sample

            dna_amount (`int`): absolute amount of DNA in the sample, for quantification

            sequences (`pd.DataFrame`): Table for each unique sequences

            sample_type (`str`, 'input' or 'reacted'): 'input' if sample is from initial pool;
                'reacted' if sample is reacted pool

            x_value (`float`): the value of time or concentration point for the sample

            metadata (`dict`): a dictionary contains metadata of the sample:

                - file_path (`str`): path to the count file

                - Other metadata extracted from the file name, or added later


    TODO:
     - add saving: to count file, to csv file, to json file
     - update docstrings (attributes, method docstrings)

    """

    def __repr__(self):
        return 'sample {}'.format(self.name)

    def __init__(self, file_path, x_value, name_pattern=None,
                 spike_in_seq=None, spike_in_amount=None, spike_in_dia=2, unit=None,
                 dna_amount=None, load_data=False,
                 silent=False):
        """`SeqSample` instance store the sequencing reads data for each sample in k-seq experiments,
        Initialize a `SeqSample` instance by linking to a read file

        **Currently only from count file is implemented**

        Args:
            file_path (`str`): directory to the read file of sample

            x_value (`float` or `str`): time/concentration/etc point value for the sample for parameter estimation.
                If `str`, it will extract value from the domain indicated by `name_pattern`

            name_pattern (`str`): optional. Pattern to automatically extract metadata using
                :func:`~k_seq.utility.extract_metadata` (click to see details). Briefly,

                - Use ``[...]`` to include the region of sample name (required),

                - Use ``{domain_name[, int/float]}`` to indicate region of domain to extract as metadata,

                including ``[,int/float]`` will convert the domain value to int/float if applicable, otherwise, string.

            load_data (`bool`): if load data from file during initializing. Recommend to set False for large files.
              Default False.

            silent (`bool`): Print progress to std out if False. Default False

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
                    Error # as data is not load yet


        """

        from .. import utility
        from pathlib import Path
        import numpy as np

        self.metadata = {}
        self._logger = utility.Logger(silent=silent)
        file_path = Path(file_path)
        self.metadata['file_path'] = str(file_path)
        self._logger.add_log("Initialize a sample from {}".format(self.metadata['file_path']))
        self.unique_seqs = None
        self.total_counts = None
        self.unit = unit
        self._silent = silent
        self._sequences = None
        if spike_in_seq is None:
            self.spike_in = None
        else:
            if isinstance(spike_in_seq, str):
                self.add_spike_in(spike_in_seq=spike_in_seq,
                                  spike_in_amount=spike_in_amount,
                                  unit=unit,
                                  spike_in_dia=spike_in_dia)
            else:
                raise ValueError('spike_in_seq needs to be a string')
        self._dna_amount = None

        if name_pattern:
            metadata = utility.extract_metadata(target=file_path.name, pattern=name_pattern)
            self.name = metadata.pop('name', None)
            self.metadata.update(metadata)
        else:
            self.name = file_path.name

        if 'input' in self.name.lower() or 'init' in self.name.lower():
            self.sample_type = 'input'
            self.x_value = np.nan
        elif isinstance(x_value, str):
            if 'input' in x_value.lower() or 'init' in x_value.lower():
                self.sample_type = 'input'
                self.x_value = np.nan
            else:
                self.sample_type = 'reacted'
                self.x_value = self.metadata[x_value]
        else:
            self.sample_type = 'reacted'
            self.x_value = x_value

        if dna_amount is not None:
            if isinstance(dna_amount, str):
                try:
                    self.dna_amount = self.metadata[dna_amount]
                except KeyError:
                    raise KeyError('Can not extract dna amount from file name')
            elif isinstance(dna_amount, float):
                self.dna_amount = dna_amount
            else:
                raise ValueError('dna_amount should be float or string')

        if load_data:
            self.load_data()

        # Import visualizers
        from ..utility import FunctionWrapper
        from .visualizer import length_dist_plot_single, sample_count_cut_off_plot_single
        self.visualizer = FunctionWrapper(data=self, functions=[length_dist_plot_single,
                                                                sample_count_cut_off_plot_single])

    @property
    def dna_amount(self):
        if self._dna_amount is None:
            if isinstance(self.spike_in, SpikeIn):
                return self.spike_in.get_dna_amount()
            else:
                raise ValueError('dna amount is not assigned and can not infer from spike in')
        else:
            return self._dna_amount

    @dna_amount.setter
    def dna_amount(self, value):
        self._dna_amount = value
        self._logger.add_log('Manually assign dna amount as {}'.format(value))

    def load_data(self):
        """Load data from file with path in ``self.matadata['file_path']``"""
        from ..data.io import read_count_file

        self._logger.add_log("Load count data from file {}".format(self.metadata['file_path']))
        self.unique_seqs, self.total_counts, self._sequences = read_count_file(self.metadata['file_path'])

    def add_spike_in(self, spike_in_seq, spike_in_amount, spike_in_dia=2, unit=None):
        self.spike_in = SpikeIn(spike_in_seq=spike_in_seq,
                                sample=self,
                                spike_in_amount=spike_in_amount,
                                spike_in_dia=spike_in_dia,
                                unit=unit)

    def summary(self):
        """Return a series as summary of sample, including:
          - sample type
          - name
          - unique seqs
          - total counts
          - x value
          - dna amount (if applicable)
          - spike in percent (if applicable)
        """
        import pandas as pd
        summary = pd.Series(data=[self.sample_type, self.name, self.unique_seqs, self.total_counts, self.x_value],
                            index=['sample type', 'name', 'unique seqs', 'total counts', 'x value'])
        if self._dna_amount is not None:
            summary = summary.append(pd.Series(data=[self._dna_amount],
                                               index=['dna amount (assigned{})'.format(
                                                   '' if self.unit is None else ', {}'.format(self.unit)
                                               )])
                                     )
        if self.spike_in is not None:
            summary = summary.append(self.spike_in.summary(verbose=False))
        return summary

    def seq_counts(self, with_spike_in=True, seq_list=None):
        if self._sequences is None:
            raise ValueError('No sample data')
        if seq_list is not None:
            return self._sequences['counts'].reindex(seq_list)
        else:
            if with_spike_in:
                return self._sequences['counts']
            else:
                return self._sequences.loc[[seq for seq in self._sequences.index if seq not in self.spike_in.members]]['counts']

    def seq_amount(self, with_spike_in=True, seq_list=None):
        if self._sequences is None:
            raise ValueError('No sample data')
        if 'amount' not in self._sequences.columns:
            self._sequences['amount'] = self._sequences['counts']/self.total_counts * self.dna_amount
        if seq_list is not None:
            return self._sequences['amount'].reindex(seq_list)
        else:
            if with_spike_in:
                return self._sequences['amount']
            else:
                return self._sequences.loc[[seq for seq in self._sequences.index if seq not in self.spike_in.members]]['amount']

    @property
    def log(self):
        """return logger value"""
        return self._logger.log


class SpikeIn(object):
    """Class to handle the spike in sequence for sample normalization

    Attributes:

        center (`str`): spike-in sequence as center seq

        sample (`SeqSample`): link to corresponding sample object

        diameter (`int`): diameter of spike-in peak account for synthesis/sequencing error

        amount (`float`): total DNA amount for the sample

        dist_to_center (`pd.Series`): a series storing edit distance of each sequence to center spike-in sequence

        members (list of `str`): list of spike-in peak members
    """

    def __repr__(self):
        return 'Spike-in peak ({})'.format(self.sample.__repr__())

    def __init__(self, spike_in_seq, sample, spike_in_amount, spike_in_dia=2, unit=None):
        self.center = spike_in_seq
        self.sample = sample
        self.amount = spike_in_amount
        self._silent = self.sample._silent
        self._diameter = spike_in_dia
        self._dist_to_center = None
        self._members = None
        self._spike_in_counts = None
        self.unit = unit
        self.sample._logger.add_log("Spike-in ({}) added, spike-in amount {}, dia={}".format(spike_in_seq,
                                                                                             spike_in_amount,
                                                                                             spike_in_dia))

    @property
    def dist_to_center(self):
        if self._dist_to_center is None:
            self._dist_to_center = self.sample.seq_counts().index.to_series().map(self._get_edit_distance)
        return self._dist_to_center

    @property
    def members(self):
        if self._members is None:
            self._members = self.dist_to_center[self.dist_to_center <= self.diameter].index
        return self._members

    @members.setter
    def members(self, value):
        self._members = value

    @property
    def spike_in_counts(self):
        if self._spike_in_counts is None:
            self._spike_in_counts = self.sample.seq_counts().loc[self.members].sum()
        return self._spike_in_counts

    @spike_in_counts.setter
    def spike_in_counts(self, value):
        self._spike_in_counts = value

    @property
    def diameter(self):
        return self._diameter

    @diameter.setter
    def diameter(self, spike_in_dia):
        if spike_in_dia != self._diameter:
            self._diameter = spike_in_dia
            self.members = self.dist_to_center[self.dist_to_center <= spike_in_dia].index
            self.spike_in_counts = self.sample.seq_counts().loc[self.members].sum()
            self.sample._logger.add_log('Set spike-in diameter as {}'.format(spike_in_dia))

    def _get_edit_distance(self, seq):
        """return edit distance of seq to the center"""
        import pandas as pd
        from Levenshtein import distance
        if isinstance(seq, pd.Series):
            seq = seq.name
        return distance(seq, self.center)

    def get_dna_amount(self, with_spike_in=True):
        """Calculate the DNA amount in the sample, by default without spike in sequences
        quant_factor (`float`): defined as
                :math:`\\frac{\\text{spike-in amount}}{\\text{total counts}\\times\\text{spike-in counts[: max_dist + 1]}}`
                if `from_spike_in_amount` is not None
                effectively the total DNA amount in the sequencing pool
        Args:
            with_spike_in (`bool`): if include spike in total DNA amount

        Returns:
            DNA amount in the unit of spike-in amount assigned, with or without spike-in sequences

        """
        spike_in_counts = self.sample.seq_counts().loc[self.members].sum()
        if with_spike_in:
            return self.amount / spike_in_counts * self.sample.total_counts
        else:
            return self.amount / spike_in_counts * (self.sample.total_counts - spike_in_counts)

    def survey_peak(self, diameter=2, accumulate=False):
        """Survey count of sequences around spike up to a given diameter (edit distance).
        Return a list of sequences within the edit distance with maximal diameter

        Args:

            diameter (`int`): the maximum distance to survey

            accumulate (`bool`): if return the accumulated count to distance i
        """

        import numpy as np
        import pandas as pd

        if not self._silent:
            print("Survey spike-in counts for {}".format(self.__repr__()))

        def get_total_counts(dist):
            if accumulate:
                return self.dist_to_center[self.dist_to_center <= dist].sum()
            else:
                return self.dist_to_center[self.dist_to_center == dist].sum()
        dist_list = np.linspace(0, diameter, diameter + 1, dtype=int)

        return pd.Series([get_total_counts(dist) for dist in dist_list], index=dist_list)

    def summary(self, verbose=False):
        """return a series of spike in info, include:
          - dna amount
          - diameter
          - spike in percent
          - seq (verbose)
          - spike-in amount (verbose)
          - spike-in counts (verbose)
        """
        import pandas as pd
        if verbose:
            return pd.Series(data=[self.get_dna_amount(), self.diameter, self.spike_in_counts/self.sample.total_counts],
                             index=[
                                 'dna amount (spike-in{})'.format('' if self.unit is None else ', {}'.format(self.unit)),
                                 'spike-in dia', 'spike-in pct'
                             ])
        else:
            return pd.Series(data=[self.get_dna_amount(), self.diameter, self.spike_in_counts/self.sample.total_counts,
                                   self.center, self.amount, self.spike_in_counts],
                             index=[
                                 'dna amount (spike-in{})'.format('' if self.unit is None else ', {}'.format(self.unit)),
                                 'spike-in dia', 'spike-in pct', 'spike-in seq', 'spike-in amount', 'spike-in counts'
                             ])


class SeqSampleSet(object):
    """Object to load and store a set of samples
    todo:
      - update methods for the new change
      - include saving - original data to json
      - add to seq table
      - update docstring
    """

    def __init__(self, x_values, file_root=None, file_list=None, file_pattern=None, black_list=None, name_pattern=None,
                 spike_in_seq=None, spike_in_amount=None, spike_in_dia=2, unit=None, dna_amount=None,
                 sort_by=None, load_data=False,
                 silent=True, note=None):
        """Initialize by linking count files under a root folder into :func:`~SeqSample` objects

        Args:
            file_root (`str`): directory to the root folder for count files

            x_values (`str` or list of `float`): Time points or concentration points values for samples. If `str`, the
              function will use it as the domain name to extract value from file name; if a list
              of `float`, it should have same length and order as sample file under `file_root`
              (Use :func:`~k_seq.utility.get_file_list' to examine the files automatically extracted)

            file_pattern (`str`): optional, name pattern to identify count files. Only file name contains
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
        from .. import utility
        import numpy as np
        import pandas as pd

        def get_file_list(file_root, file_list, file_pattern, black_list):
            """parse a list of file to import"""
            if file_root is None:
                if file_list is None:
                    raise ValueError('Please indicate either file_root or file_list')
                else:
                    return file_list
            else:
                if file_list is None:
                    file_list = [str(file) for file in Path(file_root).glob('*{}*'.format(file_pattern))
                                 if file.name not in black_list]
                    self._logger.add_log("No sample list is given, samples are collected from root folder:\n\t".format(file_root))
                else:
                    file_list = [str(Path(file_root).joinpath(file)) for file in file_list]
                return file_list

        self._logger = utility.Logger(silent=silent)
        self._logger.add_log('Dataset initialized{}'.format('' if file_root is None else ' from {}'.format(file_root)))
        self._silent = silent
        self.metadata = {
            'file_root': file_root,
            'note': note,
            'x_values': x_values,
            'black_list': black_list,
            'file_pattern': file_pattern,
            'name_pattern': name_pattern,
            'spike_in_seq': spike_in_seq,
            'spike_in_amount': spike_in_amount,
            'spike_in_dia': spike_in_dia,
            'unit': unit,
            'dna_amount': dna_amount,
            'sort_by': sort_by
        }
        if black_list is None:
            black_list = []
        file_list = get_file_list(file_root=file_root, file_list=file_list,
                                  file_pattern=file_pattern, black_list=black_list)
        self.metadata['file_list'] = file_list

        def duplicate_args(arg, arg_name):
            if isinstance(arg, list):
                if len(arg) != len(file_list):
                    raise ValueError("{} is a list, but the length does not match sample files".format(arg_name))
                else:
                    return arg
            else:
                return [arg for _ in file_list]

        x_values = duplicate_args(x_values, 'x_values')
        spike_in_seqs = duplicate_args(spike_in_seq, 'spike_in_seq')
        spike_in_amounts = duplicate_args(spike_in_amount, 'spike_in_amount')
        spike_in_dias = duplicate_args(spike_in_dia, 'spike_in_dia')
        dna_amounts = duplicate_args(dna_amount, 'dna_amount')

        self.samples = [
            SeqSample(file_path=file_path, x_value=x_value, name_pattern=name_pattern,
                      spike_in_seq=spike_in_seq, spike_in_amount=spike_in_amount,
                      spike_in_dia=spike_in_dia, dna_amount=dna_amount,
                      unit=unit, load_data=load_data, silent=silent)
            for file_path, x_value, spike_in_seq, spike_in_amount, spike_in_dia, dna_amount in zip(file_list,
                                                                                                   x_values,
                                                                                                   spike_in_seqs,
                                                                                                   spike_in_amounts,
                                                                                                   spike_in_dias,
                                                                                                   dna_amounts)
        ]
        self._logger.add_log("Samples created")

        if load_data:
            self._logger.add_log('Sample data loaded as dataset created')

        if sort_by:
            if isinstance(sort_by, str):
                sort_fn = lambda single_file: single_file.metadata[sort_by]
            elif callable(sort_by):
                sort_fn = sort_by
            else:
                raise TypeError('sort_by should either be string or callable')
            self.samples = sorted(self.samples, key=sort_fn)

        from .visualizer import count_file_info_table, count_file_info_plot, spike_in_peak_plot, rep_spike_in_plot, \
            length_dist_plot_all, sample_count_cut_off_plot_all
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

    def load_data(self):
        """Load data after creating the object, suitable for large files"""
        for sample in self.samples:
            sample.load_data()
        self._logger.add_log('Sample data loaded')

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

            silent (`bool`): don't print process if True

        """
        import numpy as np

        if quant_factors:
            # curated quant factors has highest priority
            if len(quant_factors) == len(self.sample_set):
                if isinstance(quant_factors, dict):
                    for sample in self.sample_set:
                        if sample.name in quant_factors.keys():
                            sample.quant_factor = quant_factors[sample.name]
                    else:
                        print('Warning: {} is not found in sample set'.format(sample))
                elif isinstance(quant_factors, list):
                    for sample, quant_factor in zip(self.sample_set, quant_factors):
                        sample.quant_factor = quant_factor
            else:
                raise Exception('Error: input quant_factor has different number as samples')
        elif from_spike_in_amounts is not None:
            if not hasattr(self.sample_set[0], 'spike_in'):
                if spike_in_seq is None:
                    raise Exception('Error: please provide spike-in sequence or survey the spike-in before this step')
                elif max_dist_to_survey is None:
                    max_dist_to_survey = max_dist
                    if not silent:
                        print('Use max_dist ({}) as max_dist_to_survey as it is not provided'.fromat(max_dist))
                for sample in self.sample_set:
                    sample.survey_spike_in_peak(spike_in_seq=spike_in_seq,
                                                max_dist_to_survey=max_dist_to_survey,
                                                silent=silent,
                                                inplace=True)
            if isinstance(from_spike_in_amounts, dict):
                for sample in self.sample_set:
                    if sample.name in from_spike_in_amounts.keys():
                        sample.get_quant_factor(from_spike_in_amount=from_spike_in_amounts[sample.name],
                                                max_dist=max_dist, silent=silent)
            elif isinstance(from_spike_in_amounts, list) or isinstance(from_spike_in_amounts, np.ndarray):
                for sample, spike_in_amount in zip(self.sample_set, from_spike_in_amounts):
                    sample.get_quant_factor(from_spike_in_amount=spike_in_amount,
                                            max_dist=max_dist, silent=silent)
            else:
                raise Exception('from_spike_in_amounts should be either list or dict')
        elif from_total_amounts is not None:
            if isinstance(from_total_amounts, dict):
                for sample in self.sample_set:
                    if sample.name in from_total_amounts.keys():
                        sample.get_quant_factor(from_total_amount=from_total_amounts[sample.name], silent=silent)
            elif isinstance(from_total_amounts, list) or isinstance(from_total_amounts, np.ndarray):
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
                return_df[sample.name] = sample.sequences(with_spike_in)['counts']
            return return_df

    def filter_sample(self, sample_to_keep=None, sample_to_remove=None, inplace=True):
        """filter samples in sample set

        Args:
            sample_to_keep (list of `str`): optional, names of samples to keep
            sample_to_remove (list of `str`): optional, names of samples to remove
            inplace (`bool`): return a new `SeqSampleSet` if False

        Returns: None if inplace is True, `SeqSampleSet` if inplace is False

        """

        if sample_to_keep is None and sample_to_remove is not None:
            sample_to_keep = [sample for sample in self.sample_names if sample not in sample_to_remove]
        if inplace:
            self.sample_set = [sample for sample in self.sample_set if sample.name in sample_to_keep]
        else:
            import copy
            new_set = copy.deepcopy(self)
            new_set.sample_set = [sample for sample in new_set.sample_set if sample.name in sample_to_keep]
            new_set._logger.add_log(
                'Sample is filtered and saved to this new object. Samples kept: {}'.format(
                    ','.join(sample_to_keep)
                )
            )
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