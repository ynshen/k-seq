"""Module to parse, convert, characterize count files.

Read directly from count file function was added to `data.SeqTable` thus this module is no long required in processing
count files
TODO:
  - clean up this module (after done with other data modules)
"""


class CountFile(object):

    """This class stores and handles count files get from upstream analysis in k-seq exp.

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

    Candidates:

        self.metadata = {}

        self.logger = Logger(silent=silent)

        file_path = Path(file_path)

        if not file_path.is_file():
            raise ValueError('file_path is not a valid file')

        self.metadata['file_path'] = str(file_path)
        self.logger.add("Initialize a sample from {}".format(self.metadata['file_path']))
        self.unique_seqs = None
        self.total_counts = None
        self.unit = unit
        self.silent = silent
        self.sequences = None
        self.dna_amount = None
        self.norm_factor = None

    TODO:
      - update docstrings (attributes, method docstrings)

    """

    def __repr__(self):
        return f'Sample {self.name}'

    def __init__(self, file_path, x_value=None, name_pattern=None, input_sample=None,
                 spike_in_seq=None, spike_in_amount=None, spike_in_dia=2,
                 dna_amount=None, x_unit=None, dna_unit=None, load_data=False,
                 silent=True, metadata=None, **kwargs):
        """
        Initialize a `CountFile` instance by linking it to a read file

        Args:
            file_path (`str`): full path to the read file of sample

            x_value (`float` or `str`): Controlled variable in the k-seq experiments,
                e.g. time/concentration/etc. Will be used as independent variable in parameter estimation
                If `str`, it is the name of domain indicated by `name_pattern` for automatic extraction value
                Note that only single variable is implemented

            name_pattern (`str`): optional. Pattern to automatically extract metadata using
                :func:`~k_seq.utility.file_tools.extract_metadata` (click to see details).
                Briefly,

                - Use ``[...]`` to include the region of sample name (required),

                - Use ``{domain_name[, int/float]}`` to indicate region of domain to extract as metadata,

                including ``[,int/float]`` will convert the domain value to int/float if applicable, otherwise, string.

            load_data (`bool`): if load data from file during initializing. Recommend to set False for large files.
                Default False

            silent (`bool`): Print progress to std out if False. Default False

        Example:

            .. code-block:: python

               sample = CountFile{
                   file_path = "path/to/count/file/R4B-1250A_S16_counts.txt",
                   x_value = 'byo',
                   name_pattern = "R4[{exp_rep}-{byo, float}{seq_rep}]_S{id, int}_counts.txt"
                   load_data = False,
                   silent = False
                }

            will return a `CountFile` instance `sample` that

                .. code-block:: python

                    > sample.name
                    'B-1250A'
                    > sample.x_value
                    1250.0
                    > sample.metadata
                    {
                        'exp_rep': 'B',
                        'byo': 1250.0,
                        'seq_rep': 'A',
                        'id': 16
                    }
                    > sample.sequences
                    Error # as data is not load yet
        """

        from ..utility.log import Logger
        from pathlib import Path
        import numpy as np

        self.metadata = {}
        self.logger = Logger(silent=silent)
        file_path = Path(file_path)
        if not file_path.is_file():
            raise ValueError('file_path is not a valid file')
        self.metadata['file_path'] = file_path
        self.logger.add("Initialize a sample from {}".format(self.metadata['file_path']))
        self.unique_seqs = None
        self.total_counts = None
        self.metadata['x_unit'] = x_unit
        self.metadata['dna_unit'] = dna_unit
        self.silent = silent
        self.sequences = None
        self.dna_amount = None
        self.norm_factor = None

        if name_pattern is not None:
            from ..utility.file_tools import extract_metadata
            metadata = extract_metadata(target=file_path.name, pattern=name_pattern)
            self.name = metadata.pop('name', None)
            self.metadata.update(metadata)
        else:
            self.name = file_path.name
            self.metadata.update(metadata)

        if x_value is None:
            x_value = np.nan
        if input_sample is True:
            self.sample_type = 'input'
            self.x_value = np.nan
        elif 'input' in self.name.lower() or 'init' in self.name.lower():
            # Primary: if input or init exist
            self.sample_type = 'input'
            self.x_value = np.nan
        elif isinstance(x_value, str):
            # Secondary: check from x_value
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
            # dna amount is indicated
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

        if spike_in_seq is None:
            self.spike_in = None
        else:
            if isinstance(spike_in_seq, str):
                self.add_spike_in(spike_in_seq=spike_in_seq,
                                  spike_in_amount=spike_in_amount,
                                  unit=dna_unit,
                                  spike_in_dia=spike_in_dia)
            else:
                raise ValueError('spike_in_seq needs to be a string')

        # # Import visualizers
        # from ..utility import FunctionWrapper
        # from .visualizer import length_dist_plot_single, sample_count_cut_off_plot_single
        # self.visualizer = FunctionWrapper(data=self, functions=[length_dist_plot_single,
        #                                                         sample_count_cut_off_plot_single])

    @classmethod
    def load_test_sample(cls, dataset='byo'):
        """Load test single files from BYO k-seq or BFO k-seq"""
        if dataset.lower() in ['byo']:
            kwargs = {
                'file_path': '/mnt/storage/projects/k-seq/input/byo_counts/counts-1A.txt',
                'x_value': 250,
                'spike_in_seq': 'AAAAACAAAAACAAAAACAAA',
                'spike_in_amount': 1,
                'spike_in_dia': 2,
                'x_unit': 'umol',
                'dna_unit': 'ng',
                'metadata': {}
            }
        elif dataset.lower() in ['bfo']:
            kwargs = {
                'file_path': '/mnt/storage/projects/k-seq/input/bfo_counts/counts/R4A-inputA_S1_counts.txt',
                'x_value': 'byo',
                'name_pattern': 'R4[{exp_rep}-{byo, float}{seq_rep}]_S{id, int}_counts.txt',
                'spike_in_seq': 'AAAAACAAAAACAAAAACAAA',
                'spike_in_amount': 4130,
                'x_unit': 'umol',
                'dna_unit': 'fmol'
            }
        else:
            raise NotImplementedError(f'dataset {dataset} is not implemented')
        return cls(**kwargs)

    @property
    def dna_amount(self):
        """Total DNA amount, inferred from spike-in if not assigned"""
        if self._dna_amount is None:
            if isinstance(self.spike_in, SpikeIn):
                return self.spike_in.dna_amount
            else:
                raise ValueError('No DNA amount or spike-in assigned')
        else:
            return self._dna_amount

    @dna_amount.setter
    def dna_amount(self, value):
        if value is not None:
            self._dna_amount = value
            self.norm_factor = value/self.total_counts
        else:
            self._dna_amount = None
        self.logger.add('Manually assign dna amount as {}'.format(value))

    @property
    def norm_factor(self):
        """Normalization factor for sequences, seqs DNA amount should be norm_factor * seq_counts,
        invariant for seq filtering
        """
        if self._norm_factor is None:
            if isinstance(self.spike_in, SpikeIn):
                return self.spike_in.norm_factor
            else:
                raise ValueError('No DNA amount or spike-in assigned')
        else:
            return self._norm_factor

    @norm_factor.setter
    def norm_factor(self, value):
        self._norm_factor = value

    def load_data(self):
        """Load data from file with path in ``self.matadata['file_path']``"""
        self.logger.add("Load count data from file {}".format(self.metadata['file_path']))
        self.unique_seqs, self.total_counts, self.sequences = read_count_file(self.metadata['file_path'])

    def add_spike_in(self, spike_in_seq, spike_in_amount, spike_in_dia=2, unit=None):
        self.spike_in = SpikeIn(spike_in_seq=spike_in_seq,
                                sample=self,
                                spike_in_amount=spike_in_amount,
                                spike_in_dia=spike_in_dia,
                                unit=unit)

    def seq_counts(self, remove_spike_in=True, seq_list=None, sparse=True):
        """Return raw sequence counts

        Args:
            remove_spike_in (`bool`): if remove spike-in before return data
            seq_list (list of `str`): if only return data for given sequences

        Returns:
            a `pd.Series` with counts as value and seq as index
        """
        import pandas as pd

        if self.sequences is None:
            raise ValueError('No sample data')
        if seq_list is not None:
            counts = self.sequences['counts'].reindex(seq_list)
        else:
            if remove_spike_in:
                mask = ~self.sequences.index.isin(self.spike_in.members)
                counts = self.sequences.loc[mask]['counts']
            else:
                counts = self.sequences['counts']
        if sparse:
            counts.astype(pd.SparseDtype(int, fill_value=0))
        return counts

    def seq_amount(self, remove_spike_in=True, seq_list=None, sparse=True):
        """Return absolute amount of sequences, usually not used

        Args:
            remove_spike_in (`bool`): if remove spike-in before return data
            seq_list (list of `str`): if only return data for given sequences

        Returns:
            a `pd.Series` with amount as value and seq as index

        """
        import pandas as pd
        if self.sequences is None:
            raise ValueError('No sample data')
        if 'amount' not in self.sequences.columns:
            if self.norm_factor is not None:
                self.sequences['amount'] = self.sequences['counts'] * self.norm_factor
            else:
                raise ValueError('Can not normalize counts')
        if seq_list is not None:
            amounts = self.sequences['amount'].reindex(seq_list)
        else:
            if remove_spike_in:
                mask = ~self.sequences.index.isin(self.spike_in.members)
                amounts = self.sequences.loc[mask]['amount']
            else:
                amounts = self.sequences['amount']
        if sparse:
            amounts.astype(pd.SparseDtype(float, fill_value=0))
        return amounts

    def to_series(self, verbose=False):
        """Return a series as summary of the sample, including:
          - sample type
          - name
          - x value
          - unique seqs
          - total counts
          - dna amount (if applicable)
          - *information from spike-in (if applicable, verbose choice)

        Args:
            verbose (`bool`): if return extra information in spike-in
        """
        import pandas as pd
        summary = pd.Series(data=[self.sample_type, self.name, self.x_value, self.unique_seqs, self.total_counts],
                            index=['sample type', 'name', 'x value', 'unique seqs', 'total counts'])
        if self._dna_amount is not None:
            summary = summary.append(
                pd.Series(data=[self._dna_amount],
                          index=['dna amount{}'.format(
                              '' if self.metadata['dna_unit'] is None else '({})'.format(self.metadata['dna_unit'])
                          )])
            )
        if self.sequences is not None and self.spike_in is not None:
            summary = summary.append(self.spike_in.to_series(verbose=verbose))
        return summary

    def to_dict(self, verbose=False):
        """Return a dictionary as metadata of the sample, including:
            - name
            - file_path
            - unique_seqs
            - unique_seqs_no_spike_in (if remove_spike_in is True)
            - total_counts
            - total_counts_no_spike_in (if remove_spike_in is True)
            - sample_type
            - x_value
            - dna_amount
            - dna_amount_no_spike_in (if remove_spike_in is True)
            - norm_factor
            - spike in (dict, if applicable, return members if verbose)
            - other metadata
        """
        info = {
            'name': self.name,
            'unique_seqs': self.unique_seqs,
            'total_counts': self.total_counts,
            'sample_type': self.sample_type,
            'x_value': self.x_value,
            'dna_amount_total': self.dna_amount,
            'norm_factor': self.norm_factor,
        }
        info.update(self.metadata)

        if self.spike_in is None:
            info['spike-in'] = None
        else:
            info['spike-in'] = self.spike_in.to_dict(verbose=verbose)
            info['dna_amount_no_spike_in'] = info['dna_amount_total'] - self.spike_in.spike_in_amount
            info['unique_seqs_no_spike_in'] = info['unique_seqs'] - len(self.spike_in.members)
            info['total_counts_no_spike_in'] = info['total_counts'] - self.spike_in.spike_in_counts
        return info


class SpikeIn(object):
    """Class to handle the spike in sequence for sample normalization

    Attributes:

        center (`str`): spike-in sequence as peak center

        sample (`SeqSample`): link to corresponding sample object

        radius (`int`): radius of spike-in peak, accounting for synthesis/sequencing error

        spike_in_amount (`float`): DNA amount of spike-in sequences added

        dna_amount (`float`): Total DNA amount in the sample, based on current sequences

        unit (`str`): unit of amount, e.g. ng

        norm_factor (`float`): DNA amount for a single read, amount for given read will be norm_factor * read num

        dist_to_center (`pd.Series`): edit distances for sequence to spike-in peak center

        members (list of `str`): list of spike-in peak members
    """

    def __repr__(self):
        return f'Spike-in peak for {self.sample!r}' \
               f'\n\tcenter:{self.center}' \
               f'\n\tamount: {self.spike_in_amount}, rad: {self.radius:d}, unit: {self.unit}'

    def __init__(self, spike_in_seq, sample, spike_in_amount=None, unit=None, spike_in_dia=2):
        self.center = spike_in_seq
        self.sample = sample
        self.spike_in_amount = spike_in_amount
        self._silent = self.sample.silent
        self.radius = spike_in_dia
        self.unit = unit
        self.dist_to_center = None
        self.members = None
        self.spike_in_counts = None
        self.sample.logger.add("Spike-in ({}) added, spike-in amount {}, rad {}, unit {}".format(
            spike_in_seq,
            spike_in_amount,
            spike_in_dia,
            unit
        ))

    def _get_edit_distance(self, seq):
        """use `python-levenshtein` to calculate edit distances of seqs to the center"""
        import pandas as pd
        from Levenshtein import distance
        if isinstance(seq, pd.Series):
            seq = seq.name
        return distance(seq, self.center)

    @property
    def dist_to_center(self):
        """edit distances for sequence to spike-in peak center, only calculate once"""
        if self._dist_to_center is None:
            self._dist_to_center = self.sample.sequences.index.to_series().map(self._get_edit_distance)
        return self._dist_to_center

    @dist_to_center.setter
    def dist_to_center(self, value):
        """Should not assign manual except for release memory"""
        self._dist_to_center = value

    @property
    def members(self):
        """A list of sequences considered as in the peak, change as radius change"""
        if self._members is None:
            self._members = self.dist_to_center[self.dist_to_center <= self.radius].index.to_series().values
        return self._members

    @members.setter
    def members(self, value):
        """it can be manually set"""
        self._members = value

    @property
    def spike_in_counts(self):
        """Total counts of sequences in `self.members`, updated as radius change"""
        if self._spike_in_counts is None:
            self._spike_in_counts = self.sample.sequences.loc[self.members]['counts'].sum()
        return self._spike_in_counts

    @spike_in_counts.setter
    def spike_in_counts(self, value):
        self._spike_in_counts = value

    @property
    def radius(self):
        return self._diameter

    @radius.setter
    def radius(self, spike_in_dia):
        """Update members and spike_in_counts when updating radius"""
        if not hasattr(self, '_diameter'):
            self._diameter = spike_in_dia
        else:
            if spike_in_dia != self._diameter:
                self._diameter = spike_in_dia
                self.members = self.dist_to_center[self.dist_to_center <= spike_in_dia].index.to_series().values
                self.spike_in_counts = self.sample.sequences.loc[self.members]['counts'].sum()
                self.sample.logger.add('Set spike-in radius to {}'.format(spike_in_dia))

    @property
    def norm_factor(self):
        """
        norm_factor (`float`): defined as
          :math:`\\frac{\\text{spike-in amount}}{\\text{spike-in counts}`
        """
        return self.spike_in_amount/self.spike_in_counts

    @property
    def dna_amount(self):
        """Total DNA amount in the sample, calculated on current sequences in the sample
        """
        return self.norm_factor * self.sample.total_counts

    def survey_peak(self, radius=2, accumulate=False):
        """Survey count of sequences around spike up to a given radius (edit distance).

        Args:
            radius (`int`): the maximum distance to survey
            accumulate (`bool`): if return the accumulated count to distance i

        Returns:
            A `pd.Series` of [accumulated] seq counts at different edit distance
        """

        import numpy as np
        import pandas as pd

        if not self._silent:
            print("Survey spike-in counts for {}".format(self.__repr__()))

        def get_total_counts(dist):
            if accumulate:
                return self.sample.sequences['counts'][self.dist_to_center <= dist].sum()
            else:
                return self.sample.sequences['counts'][self.dist_to_center == dist].sum()
        dist_list = np.linspace(0, radius, radius + 1, dtype=int)
        return pd.Series([get_total_counts(dist) for dist in dist_list], index=dist_list)

    def to_series(self, verbose=False):
        """return a series of spike in info, including:
          - dna amount (from spike-in, [unit])
          - spike-in rad: radius
          - spike-in pct: percent
          - spike-in seq (verbose)
          - spike-in amount (verbose)
          - spike-in counts (verbose)

          Args:
              verbose (`bool`): if include extra information
        """
        import pandas as pd
        if not verbose:
            return pd.Series(data=[self.dna_amount, self.radius, self.spike_in_counts / self.sample.total_counts],
                             index=[
                                 'dna amount (from spike-in{})'.format('' if self.unit is None else ', {}'.format(self.unit)),
                                 'spike-in rad', 'spike-in pct'
                             ])
        else:
            return pd.Series(data=[self.dna_amount, self.radius, self.spike_in_counts / self.sample.total_counts,
                                   self.center, self.spike_in_amount, self.spike_in_counts],
                             index=[
                                 'dna amount (from spike-in{})'.format('' if self.unit is None else ', {}'.format(self.unit)),
                                 'spike-in rad', 'spike-in pct', 'spike-in seq', 'spike-in amount', 'spike-in counts'
                             ])

    def to_dict(self, verbose=False):
        """Return necessary info of spike-in as a dictionary, including:
          - seq: spike-in seq
          - radius:
          - dna_amount: DNA amount of sample pool calculated by spike-in
          - norm_factor
          - unit
          - members (verbose): list

        Args:
            verbose (`bool`): if return extra info (members)
        """
        info = {
            'seq': self.center,
            'radius': self.radius,
            'dna_amount': self.dna_amount,
            'norm_factor': self.norm_factor,
            'unit': self.unit
        }
        if verbose:
            info.update({'members': self.members})

        return info


def read_count_file(file_path, as_dict=False, number_only=False):
    """Read a single count file generated from Chen lab's customized scripts

    Count file format:
    ::
        number of unique sequences = 2825
        total number of molecules = 29348173

        AAAAAAAACACCACACA               2636463
        AATATTACATCATCTATC              86763
        ...

    Args:
        file_path (str): full directory to the count file
        as_dict (bool): return a dictionary instead of a `pd.DataFrame`
        number_only (bool): only return number of unique seqs and total counts if True

    Returns:
        unique_seqs (`int`): number of unique sequences in the count file
        total_counts (`int`): number of total reads in the count file
        sequence_counts (`pd.DataFrame`): with `sequence` as index and `counts` as the first column
    """
    import pandas as pd

    with open(file_path, 'r') as file:
        unique_seqs = int([elem for elem in next(file).strip().split()][-1])
        total_counts = int([elem for elem in next(file).strip().split()][-1])
        if number_only:
            sequence_counts = None
            as_dict = True
        else:
            next(file)
            sequence_counts = {}
            for line in file:
                seq = line.strip().split()
                sequence_counts[seq[0]] = int(seq[1])

    if as_dict:
        return unique_seqs, total_counts, sequence_counts
    else:
        return unique_seqs, total_counts, pd.DataFrame.from_dict(sequence_counts, orient='index', columns=['counts'])


class CountFileSet(object):
    """Object to load and store a set of samples
    todo:
      - add to seq table
      - update docstring
    """

    def __repr__(self):
        return f"Count files from {len(self.samples)} samples\n\tNote: {self.metadata['note']}"

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.samples[item]
        elif isinstance(item, str):
            return self._sample_indexer[item]
        elif isinstance(item, list):
            if isinstance(item[0], str):
                return [self._sample_indexer[key] for key in item]
        raise TypeError('Index should be either int, string (sample name), or list of string')

    def __init__(self, file_root, x_values=None, input_samples=None, file_list=None, file_pattern=None,
                 black_list=None, name_pattern=None,
                 spike_in_seq=None, spike_in_amount=None, spike_in_dia=2, x_unit=None, dna_unit=None, dna_amount=None,
                 sort_by=None, load_data=False, silent=True, metadata=None, note=None):
        """
        todo:
          update this docstring
          survey files first, add x_values later

        Initialize by linking count files under a root folder into :func:`~CountFile` objects


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
        from ..utility.log import Logger
        from ..utility.file_tools import get_file_list
        from ..utility.func_tools import param_to_dict

        self.logger = Logger(silent=silent)
        self.logger.add('Dataset initialized{}'.format('' if file_root is None else ' from {}'.format(file_root)))
        self.silent = silent

        file_list = get_file_list(file_root=file_root, file_list=file_list,
                                  pattern=file_pattern, black_list=black_list, full_path=True)
        self.file_list = file_list
        file_names = [file.name for file in file_list]

        if sort_by is not None:
            # pre-run sample creation to extract sample info and sort samples
            sample_params = param_to_dict(key_list=file_names,
                                          file_path=file_list, x_value=x_values, name_pattern=name_pattern,
                                          spike_in_seq=None, spike_in_amount=None,
                                          spike_in_dia=None, dna_amount=dna_amount, x_unit=None,
                                          dna_unit=None, load_data=False, silent=True, metadata=metadata)
            self.samples = [CountFile(**args) for args in sample_params.values()]
            self.sort_sample(sort_by=sort_by)
            file_list = [sample.metadata['file_path'] for sample in self.samples]
            file_names = [file.name for file in file_list]

        sample_params = param_to_dict(key_list=file_names,
                                      file_path=file_list, x_value=x_values, name_pattern=name_pattern,
                                      spike_in_seq=spike_in_seq, spike_in_amount=spike_in_amount,
                                      spike_in_dia=spike_in_dia, dna_amount=dna_amount, x_unit=x_unit,
                                      dna_unit=dna_unit, load_data=load_data, silent=silent, metadata=metadata)

        if input_samples is not None:
            for sample, params in sample_params.items():
                params['input_sample'] = sample in input_samples

        self.samples = [CountFile(**args) for args in sample_params.values()]

        self._sample_indexer = {sample.name: sample for sample in self.samples}
        self._sample_indexer.update({sample.metadata['file_path'].name: sample for sample in self.samples})
        self.logger.add("Samples created")

        if load_data:
            self.logger.add('Sample data loaded as dataset created')

        # todo: clear logic of why need metadata and what to record in metadata, avoiding confusion
        from datetime import datetime
        self.metadata = {
            'file_root': file_root,
            'time_created': datetime.now(),
            'note': note,
            'black_list': black_list,
            'file_pattern': file_pattern,
            'name_pattern': name_pattern,
            'x_unit': x_unit,
            'sort_by': sort_by
            }

        # from .visualizer import count_file_info_table, count_file_info_plot, spike_in_peak_plot, rep_spike_in_plot, \
        #     length_dist_plot_all, sample_count_cut_off_plot_all
        # from ..utility import FunctionWrapper
        # self.visualizer = FunctionWrapper(data=self,
        #                                   functions=[
        #                                       count_file_info_table,
        #                                       count_file_info_plot,
        #                                       spike_in_peak_plot,
        #                                       rep_spike_in_plot,
        #                                       length_dist_plot_all,
        #                                       sample_count_cut_off_plot_all
        #                                   ])

    @property
    def sample_names(self):
        return [sample.name for sample in self.samples]

    @staticmethod
    def check_file_list(file_root, file_list=None, file_pattern=None, black_list=None, full_path=True, **kwargs):
        """Utility function to list the files will be imported"""
        from ..utility.file_tools import get_file_list
        file_list = get_file_list(file_root=file_root, file_list=file_list,
                                  pattern=file_pattern, black_list=black_list, full_path=full_path)
        return file_list

    def sort_sample(self, sort_by):
        """Sort sample order by attribute name, metadata name, or a callable on `CountFile` object"""
        if isinstance(sort_by, str):
            def sort_fn(single_file):
                if hasattr(single_file, sort_by):
                    return getattr(single_file, sort_by)
                else:
                    return single_file.metadata[sort_by]
        elif callable(sort_by):
            sort_fn = sort_by
        else:
            raise TypeError('sort_by should either be string or callable')
        self.samples = sorted(self.samples, key=sort_fn)

    def load_data(self):
        """Load data after creating the object, suitable for large files"""
        for sample in self.samples:
            sample.load_data()
        self.logger.add('Sample data loaded')

    def add_dna_amount(self, dna_amounts, unit=None):
        if isinstance(dna_amounts, list):
            if len(dna_amounts) == len(self.samples):
                for sample, dna_amount in zip(self.samples, dna_amounts):
                    sample.dna_amount = dna_amount
                    sample.metadata['dna_unit'] = unit
                self.metadata['unit'] = unit
            else:
                raise ValueError('Length of dna_amounts is different from sample number')
        elif isinstance(dna_amounts, dict):
            for key, dna_amount in dna_amounts.items():
                if key in self.sample_names:
                    self._sample_indexer[key].dna_amount = dna_amount
                    self._sample_indexer[key].unit = unit
                else:
                    raise Warning('{} not found in sample set'.format(key))
        else:
            raise TypeError('dna_amounts should be either list or dictionary')

    def add_spike_in(self, spike_in_seq=None, spike_in_amount=None, spike_in_dia=2, unit=None):
        """
        Add spike-in sequences for all samples, for each parameter, single value or dictionary could be passed,
        if a dictionary is passed, the key should be sample name and explicitly list parameters for all samples
        """
        sample_keys = [sample.name for sample in self.samples]
        from ..utility.func_tools import param_to_dict
        param_dict = param_to_dict(key_list=sample_keys,
                                   spike_in_seq=spike_in_seq, spike_in_amount=spike_in_amount, spike_in_dia=spike_in_dia,
                                   unit=unit)
        for key, args in param_dict.items():
            self._sample_indexer[key].add_spike_in(**args)

    def summary(self):
        import pandas as pd
        return pd.DataFrame(data=[sample.to_series() for sample in self.samples])

    def to_SeqTable(self, sample_list=None, black_list=None, seq_list=None,
                    remove_spike_in=True, return_counts=True, note=None):
        """Convert to a `SeqTable` object
        todo: finish this
        Args:
            remove_spike_in (`bool`): remove spike in if True. Default True.
            note (`str`): optional. Note about the sample set

        Returns: `SeqTable` instance
        """
        from .seq_table import SeqTable

        if sample_list is None:
            sample_list = self.sample_names

        if black_list is not None:
            sample_list = [sample for sample in sample_list if sample not in black_list]

        if seq_list is None:
            seq_table = self.to_dataframe(samples=sample_list,
                                          remove_spike_in=remove_spike_in,
                                          return_counts=return_counts)
        else:
            seq_table = self.to_dataframe(samples=sample_list,
                                          seq_list=seq_list,
                                          remove_spike_in=remove_spike_in,
                                          return_counts=return_counts)

        # Collect metadata
        sample_meta = self.sample_into_to_dict(sample_list=sample_list, verbose=False)

        # Init Grouper
        spike_in_seqs = []
        for sample in self.samples:
            if sample.spike_in is not None:
                spike_in_seqs += list(sample.spike_in.members)
        spike_in_seqs = list(set(spike_in_seqs))
        grouper = {'input': [sample for sample in sample_list if self[sample].sample_type == 'input'],
                   'reacted': [sample for sample in sample_list if self[sample].sample_type == 'reacted']}
        if spike_in_seqs != []:
            grouper['spike_in_seqs'] = {'axis':0, 'group': spike_in_seqs}

        if note is None:
            note = self.metadata['note']

        from .seq_table import SeqTable
        return SeqTable(data_mtx=seq_table, data_unit='count', grouper=grouper,
                        seq_metadata=None, sample_metadata=sample_meta, x_unit=self.metadata['x_unit'],
                        dataset_metadata=None, note=note)

    def to_dataframe(self, samples=None, seq_list=None, remove_spike_in=True, return_counts=True, sparse=True):
        """Return a dataframe within index as sequences, columns as samples"""

        import pandas as pd

        if samples is None:
            samples = self.sample_names
        if return_counts:
            if sparse:
                dtype = pd.SparseDtype('int', fill_value=0)
            else:
                dtype = int
            return pd.DataFrame({
                sample: self[sample].seq_counts(remove_spike_in=remove_spike_in, seq_list=seq_list, sparse=False)
                for sample in samples
            }, columns=samples).fillna(0, inplace=False).astype(dtype)
        else:
            if sparse:
                dtype = pd.SparseDtype('float', fill_value=0.0)
            else:
                dtype = float
            return pd.DataFrame({
                sample: self[sample].seq_amount(remove_spike_in=remove_spike_in, seq_list=seq_list, sparse=False)
                for sample in samples
            }, columns=samples).fillna(0.0, inplace=False).astype(dtype)

    def sample_into_to_dict(self, sample_list=None, exclude_spike_in=False, verbose=False):
        if sample_list is None:
            sample_list = self.sample_names
        return {self[sample].name: self[sample].to_dict(verbose=verbose) for sample in sample_list}

    def to_csv(self, file_path, samples=None, seq_list=None, remove_spike_in=False, return_counts=False):
        self.to_dataframe(samples=samples, seq_list=seq_list,
                          remove_spike_in=remove_spike_in, return_counts=return_counts).to_csv(file_path)

    # def filter_sample(self, sample_to_keep=None, sample_to_remove=None, inplace=False):
    #     """filter samples in sample set
    #
    #     Args:
    #         sample_to_keep (list of `str`): optional, names of samples to keep
    #         sample_to_remove (list of `str`): optional, names of samples to remove
    #         inplace (`bool`): return a new `SeqSampleSet` if False
    #
    #     Returns: None if inplace is True, `SeqSampleSet` if inplace is False
    #
    #     """
    #
    #     if sample_to_keep is None and sample_to_remove is not None:
    #         sample_to_keep = [sample for sample in self.sample_names if sample not in sample_to_remove]
    #
    #     if inplace:
    #         self.samples = [sample for sample in self.samples if sample.name in sample_to_keep]
    #     else:
    #         import copy
    #         new_set = copy.deepcopy(self)
    #         new_set.samples = [sample for sample in new_set.samples if sample.name in sample_to_keep]
    #         new_set._logger.add(
    #             'Sample is filtered and saved to this new object. Samples kept: {}'.format(
    #                 ','.join(sample_to_keep)
    #             )
    #         )
    #         return new_set

    def to_pickle(self, file_path):
        import pickle
        with open(file_path, 'wb') as handle:
            pickle.dump(obj=self, file=handle, protocol=-1)

    @staticmethod
    def from_pickle(file_path):
        import pickle
        with open(file_path) as handle:
            return pickle.load(handle)

    @classmethod
    def load_default(cls, dataset='byo'):
        """Load dataset from BYO k-seq or BFO k-seq"""
        import numpy as np
        if dataset.lower() in ['byo']:
            kwargs = {
                'file_root': '/mnt/storage/projects/k-seq/input/byo_counts/',
                'x_values': np.repeat([250, 50, 10, 2], repeats=6),
                'file_list': None,
                'file_pattern': 'counts-',
                'black_list': None,
                'name_pattern': 'counts-[{bfo_group}{exp_rep}.txt]',
                'spike_in_seq': 'AAAAACAAAAACAAAAACAAA',
                'spike_in_amount': np.repeat([1, 0.4, 0.3, 0.1], repeats=6),
                'spike_in_dia': 2,
                'x_unit': 'umol',
                'dna_unit': 'ng',
                'dna_amount':None,
                'sort_by': 'name',
                'load_data': False,
                'silent': True,
                'metadata': None,
                'note': 'BYO k-seq data by Abe Pressman'
            }
        elif dataset.lower() in ['bfo']:
            kwargs = {
                'file_root': '/mnt/storage/projects/k-seq/input/bfo_counts/counts/',
                'x_values': 'byo',
                'name_pattern': 'R4[{exp_rep}-{byo, float}{seq_rep}]_S{id, int}_counts.txt',
                'spike_in_seq': 'AAAAACAAAAACAAAAACAAA',
                'file_list': None,
                'file_pattern': '.txt',
                'black_list': None,
                'spike_in_amount': np.tile([4130, 1240, 826, 413, 207, 82.6, 41.3], reps=4),
                'spike_in_dia': 2,
                'x_unit': 'umol',
                'dna_unit': 'fmol',
                'dna_amount': None,
                'sort_by': 'id',
                'load_data': False,
                'silent': True,
                'metadata': None,
                'note': 'BFO k-seq data by Evan Jensen'
            }
        elif dataset.lower() in ['byo-doped']:
            kwargs = {
                'file_root': '/mnt/storage/projects/k-seq/input/byo_doped/counts',
                'x_values': np.concatenate((
                    np.repeat([1250, 250, 50, 10, 2], repeats=3) * 1e-6,
                    np.array([np.nan])), axis=0
                ),
                'file_list': None,
                'file_pattern': 'counts-',
                'black_list': None,
                'input_samples': 'counts-d-R0.txt',
                'name_pattern': 'counts-d-[{byo}{exp_rep}].txt',
                'spike_in_seq': 'AAAAACAAAAACAAAAACAAA',
                'spike_in_amount': np.concatenate((
                    np.repeat([2, 2, 1, 0.2, .04], repeats=3),
                    np.array([10])), axis=0
                ),
                'spike_in_dia': 4,
                'x_unit': 'mol',
                'dna_unit': 'ng',
                'dna_amount': None,
                'sort_by': 'name',
                'load_data': False,
                'silent': True,
                'metadata': None,
                'note': 'BYO doped-pool k-seq data by Abe Pressman'
            }
        else:
            raise NotImplementedError(f'dataset {dataset} is not implemented')
        return cls(**kwargs)




