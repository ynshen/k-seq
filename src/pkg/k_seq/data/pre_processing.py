"""This module contains methods for data preprocessing from count files to ``SequenceSet`` for fitting
TODOs:
  - write output function for each class as JSON file
"""

import numpy as np
import pandas as pd
from . import io
from .. import utility


class SequencingSample(object):
    """This class describes experimental samples sequenced in k-seq
    """

    def __init__(self, file_dirc, x_value, silent=True, name_pattern=None):
        """``SequencingSample`` instance store the data of each sequencing sample in k-seq experiment
        Initialize a ``SequencingSample`` instance by reading a count file

        Args:
            file_dirc (str): directory to the count file of sample

            x_value (float or str): corresponding x_value for the sample. If string, it will extract value from
                corresponding domain with the name

            silent (bool): optional. Print progress to std.out if True

            name_pattern (str): optional. Pattern to extract metadata using :func:`~k_seq.utility.extract_metadata`.
                Use ``[...]`` to include the region of sample_name,
                use ``{domain_name[, int/float]}`` to indicate region of domain to extract as metadata,
                including ``[,int/float]`` will convert the domain value to float in applicable, otherwise, string.

        Example:

            Example on metadata extraction from pattern:

                .. code-block:: python

                   SequencingSample{
                       sample_name = "R4B-1250A_S16_counts.txt"
                       pattern = "R4[{exp_rep}-{concentration, float}{seq_rep}_S{id, int}_counts.txt"
                       ...
                    }

            will return

                .. code-block:: python

                   SequencingSample.metadata = {
                       'exp_rep': 'B',
                       'concentration': 1250.0,
                       'seq_rep': 'A',
                       'id': 16
                    }

        Attributes:

            name (str): name of sample, extracted from file name

            unique_seqs (int): number of unique sequences in the sample

            total_counts (int): number of total reads in the sample

            sequences (dict): a dictionary of counts for each unique sequences
                {seq: counts}

            sample_type ('input' or 'reacted'): 'input' if sample is initial pool; 'reacted' if sample is reacted pool

            x_value (float): the value of time point or concentration point for the sample

            metadata (dict): a dictionary contains metadata of the sample:

                - file_dirc (str):

                - time_stamp (time): time the instance created

                - Other metadata extracted from file name
        """

        import datetime
        from .. import utility

        self.metadata = {}
        sample_name = file_dirc[file_dirc.rfind('/') + 1:]
        self.metadata['file_dirc'] = file_dirc
        self.unique_seqs, self.total_counts, self.sequences = io.read_count_file(self.metadata['file_dirc'])

        if name_pattern:
            metadata = utility.extract_metadata(target=sample_name, pattern=name_pattern)
            self.name = metadata.pop('name', None)
            self.metadata.update(metadata)
        else:
            self.name = sample_name

        if 'input' in self.name or 'Input' in self.name:
            self.sample_type = 'input'
        else:
            self.sample_type = 'reacted'

        if self.sample_type == 'input':
            self.x_value = np.nan
        else:
            if type(x_value) == str:
                self.x_value = self.metadata[x_value]
            else:
                self.x_value = x_value

        self.metadata['timestamp'] = str(datetime.datetime.now())
        if not silent:
            print("Sample {} imported from {}".format(self.name, self.metadata['file_dirc']))

    def survey_spike_in(self, spike_in, max_dist_to_survey=10, silent=True, inplace=True):
        """Survey spike-in counts in the sample.
        Count number of sequences at different edit distance to spike-in sequences (external standard for quantification)
          in the sample. Add attribute ``spike_in`` to the instance if inplace is True.

        Args:
            spike_in (str): the sequence of spike-in, as the center sequence

            max_dist_to_survey (int): the maximum distance to survey

            silent (bool): don't print progress if True

            inplace (bool): Add attribute ``spike_in`` if inplace is True, else return a dict object
                {
                    'spike_in_seq': str, spike in sequence,
                    'spike_in_counts': list of int, number of sequences that is i edit distance away
                }

        Attributes:

            spike_in (dict):

                - spike_in_counts (list): length max_dist_to_survey + 1, number of total counts with different edit
                  distance to spike-in sequence

                - spike_in (str): center spike_in sequence

        """
        import Levenshtein

        results = dict()
        results['spike_in_counts'] = np.zeros(max_dist_to_survey + 1, dtype=np.int)
        results['spike_in_seq'] = spike_in
        for seq,counts in self.sequences.items():
            dist = Levenshtein.distance(spike_in, seq)
            if dist <= max_dist_to_survey:
                results['spike_in_counts'][dist] += counts
        if not silent:
            print("Survey spike-in counts for sample {}. Done.".format(self.name))
        if inplace:
            self.spike_in = results
        else:
            return results

    def get_quant_factor(self, spike_in_amount, max_dist=0, silent=True):
        """Calculate quantification factor for the sample.
        Add ``quant_factor`` and ``quant_factor_max_dist`` attributes to the instance

        Args:
            max_dist (int): maximum edit distance for a sequence to be pike-in

            spike_in_amount (float): amount of actual spike-in added in experiment

        Attributes:
            quant_factor (float): defined as :math:`\\frac{\\text{spike-in amount}}{\\text{total counts}\\times\\text{spike-in counts[: max_dist + 1]}}`

            quant_factor_max_dist (int): maximum edit distance for a sequence to be spike-in

        """

        self.quant_factor = spike_in_amount * self.total_counts / np.sum(self.spike_in['spike_in_counts'][:max_dist + 1])
        self.spike_in['quant_factor_max_dist'] = max_dist
        self.spike_in['spike_in_amount'] = spike_in_amount

        if not silent:
            print("Calculate quant-factor for sample {}. Done.".format(self.name))


class SequencingSampleSet(object):
    """Object to load and store batch of samples
    """

    def __init__(self, file_root, x_values, sample_list=None, pattern=None, name_pattern=None,
                 sort_fn=None, black_list=[], silent=True):
        """load count files under a root folder into :func:`~SequencingSample` objects

        Args:
            file_root (str): directory to the root folder

            x_values (str or list of float): string or a list of floats. The time points or concentration points value for
              each sample. If string, the function will use it as domain name to extract x_value from file name; if a list
              of floats, it should have same length and order as sample file under file_root
              (Use :func:`~k_seq.utility.get_file_list' to examine the files automatically extracted)

            sample_list (list of str): optional, only import the file in sample_list if not None

            pattern (str): optional, file name pattern to identify count files. Only file with name strictly contains
              the pattern will be collected.

            name_pattern (str): optional. Pattern to extract metadata, see :func:`~k_seq.utility.extract_metadata'

            sort_fn (callable): optional. A callable to customize sample order

            black_list (list of str): name of sample files that will be excluded in loading

            silent (boolean): don't print process if True

        """
        if sample_list is None:
            sample_list = utility.get_file_list(file_root=file_root, pattern=pattern)
            if not silent:
                print("NOTICE: no sample_list is given, samples will extract automatically from file_root.")
            if type(x_values) != str:
                raise Exception("No sample_list is given, "
                                "please indicate domain name instead of list of real values to extract x_values")
        self.sample_set = []
        if file_root[-1] != '/':
            file_root += '/'
        if type(x_values) == str:
            x_values = [x_values for _ in sample_list]
        for sample_ix, sample_name in enumerate(sample_list):
            if sample_name not in black_list:
                sample = SequencingSample(file_dirc=file_root + sample_name,
                                          x_value=x_values[sample_ix],
                                          name_pattern=name_pattern,
                                          silent=silent)
                self.sample_set.append(sample)
        if sort_fn:
            self.sample_set.sort(key=sort_fn)

        self.sample_num = len(self.sample_set)
        self.sample_names = [sample.name for sample in self.sample_set]

        if not silent:
            print("Samples imported from {}".format(file_root))

    def get_quant_factors(self, spike_in_amounts, spike_in='AAAAACAAAAACAAAAACAAA', max_dist=2, max_dist_to_survey=None,
                          quant_factor=None, survey_only=False, silent=True):
        """Calculate quantification factors for each sample in SequencingSampleSet.
        This method will first survey the spike-in sequence, then calculate the quantification factor for each sample if
        applies.

        Args:
            spike_in_amounts (list/dict of float): a list/dict of absolute amount of spike-in sequence for each sample
              if dict, the key must match the name of sample.

            spike_in (str): optional, spike in sequence. Default: AAAAACAAAAACAAAAACAAA

            max_dist (int): optional, maximum edit distance to center spike-in sequence to be counted as spike-in.
              Default: 2

            max_dist_to_survey (int): optional, survey the spike-in counts with maximum edit distance if not None

            quant_factor (list/dict of float): optional, accept a list/dict of manually applied quant_factor

            survey_only (bool): optional. If True, only survey the spike-in counts within maximum edit distance

            silent (bool): don't print process if True

        """

        for sample in self.sample_set:
            sample.survey_spike_in(spike_in=spike_in, max_dist_to_survey=max_dist_to_survey,
                                   silent=silent, inplace=True)
        if not survey_only:
            if quant_factor:
                for sample_ix, sample in enumerate(self.sample_set):
                    if isinstance(quant_factor, dict):
                        if sample.name in quant_factor.keys():
                            sample.quant_factor = quant_fasctor[sample.name]
                    elif isinstance(quant_factor, list):
                        sample.quant_factor = quant_factor[sample_ix]
            else:
                for sample_ix, sample in enumerate(self.sample_set):
                    if isinstance(spike_in_amounts, dict):
                        if sample.name in spike_in_amounts.keys():
                            sample.get_quant_factor(spike_in_amount=spike_in_amounts[sample.name],
                                                    max_dist=max_dist)
                    elif isinstance(spike_in_amounts, list):
                        sample.get_quant_factor(spike_in_amount=spike_in_amounts[sample_ix],
                                                max_dist=max_dist)

    def filter_sample(self, sample_to_keep, inplace=True):
        """filter samples in sample set

        Args:
            sample_to_keep (list of str): name of samples to keep
            inplace (bool): return a new SequencingSampleSet if False

        Returns: None if inplace is True, SequencingSampleSet if inplace is False

        """

        if inplace:
            self.sample_set = [sample for sample in self.sample_set if sample.name in sample_to_keep]
            self.sample_num = len(self.sample_set)
            self.sample_names = sample_to_keep
        else:
            import copy
            new_set = copy.deepcopy(self)
            new_set.sample_set = [sample for sample in new_set.sample_set if sample.name in sample_to_keep]
            new_set.sample_num = len(new_set.sample_set)
            new_set.sample_names = sample_to_keep
            return new_set


class SequenceSet(object):
    """This class contains the dataset of valid sequences extracted and aligned from a list of ``SequencingSample``
    """

    def __init__(self, sample_set, remove_spike_in=True, note=None):
        """Initialize from a list of ``SequencingSample`` to a SequenceSet object.
        Find all valid sequences that occur at least once in any 'input' sample and once in any 'reacted' sample

        Args:

            sample_set (``SequencingSampleSet``): valid samples to convert to ``SequenceSet``

            remove_spike_in (bool): sequences considered as spike-in will be removed, all number are calculated after
              removal

            note (str): optional. Additional note to add to the dataset


        Attributes:
`
            dataset_info (dict): dictionary of basic info of dataset:

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
        import Levenshtein
        import datetime

        # find valid sequence set
        input_seq_set = set()
        reacted_seq_set = set()

        sample_set = sample_set.sample_set

        if remove_spike_in:
            for sample in sample_set:
                if sample.sample_type == 'input':
                    input_seq_set.update([
                        seq for seq in sample.sequences.keys()
                        if Levenshtein.distance(seq, sample.spike_in['spike_in_seq']) > sample.spike_in['quant_factor_max_dist']
                    ])
                elif sample.sample_type == 'reacted':
                    reacted_seq_set.update([
                        seq for seq in sample.sequences.keys()
                        if Levenshtein.distance(seq, sample.spike_in['spike_in_seq']) > sample.spike_in['quant_factor_max_dist']
                    ])
        else:
            for sample in sample_set:
                if sample.sample_type == 'input':
                    input_seq_set.update(list(sample.sequences.keys()))
                elif sample.sample_type == 'reacted':
                    reacted_seq_set.update(list(sample.sequences.keys()))

        valid_set = input_seq_set & reacted_seq_set
        self.dataset_info = {
            'input_seq_num': len(input_seq_set),
            'reacted_seq_num': len(reacted_seq_set),
            'valid_seq_num': len(valid_set),
            'remove_spike_in': remove_spike_in
        }
        if note:
            self.dataset_info['note'] = note

        # preserve sample info
        self.sample_info = {}
        for sample in sample_set:
            sample_info_dict = sample.__dict__.copy()
            sequences = sample_info_dict.pop('sequences', None)
            self.sample_info[sample.name] = {
                'valid_seqs_num': np.sum([1 for seq in sequences.keys() if seq in valid_set]),
                'valid_seqs_counts': np.sum([seq[1] for seq in sequences.items() if seq[0] in valid_set])
            }
            self.sample_info[sample.name].update(sample_info_dict)

        # create valid sequence table
        self.count_table = pd.DataFrame(index=list(valid_set), columns=[sample.name for sample in sample_set])
        for seq in valid_set:
            for sample in sample_set:
                if seq in sample.sequences.keys():
                    self.count_table.loc[seq, sample.name] = sample.sequences[seq]

        self.dataset_info['timestamp'] = str(datetime.datetime.now())

    def get_reacted_frac(self, input_average='median', black_list=None, inplace=True):
        """Calculate reacted fraction for sequences

        Args:

            input_average ('median' or 'mean'): method to calculate the average amount of input for a sequence

            black_list (list of str): optional, list of names of samples to be excluded in calculation

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
        input_samples = [sample[0] for sample in self.sample_info.items()
                         if sample[0] not in black_list and sample[1]['sample_type'] == 'input']
        reacted_samples = [sample[0] for sample in self.sample_info.items()
                           if sample[0] not in black_list and sample[1]['sample_type'] == 'reacted']
        reacted_frac_table = pd.DataFrame(index=self.count_table.index, columns=reacted_samples)
        reacted_frac_table.input_avg_type = input_average
        reacted_frac_table.col_x_values = [float(self.sample_info[sample]['x_value']) for sample in reacted_frac_table.columns]

        if input_average == 'median':
            input_amount_avg = np.nanmedian(np.array([
                list(self.count_table[sample] / self.sample_info[sample]['total_counts'] *
                     self.sample_info[sample]['quant_factor'])
                for sample in input_samples
            ]), axis=0)
        elif input_average == 'mean':
            input_amount_avg = np.nanmean(np.array([
                list(self.count_table[sample] / self.sample_info[sample]['total_counts'] *
                     self.sample_info[sample]['quant_factor'])
                for sample in input_samples
            ]), axis=0)
        else:
            raise Exception("Error: input_average should be 'median' or 'mean'")

        reacted_frac_table.input_avg = input_amount_avg
        for sample in reacted_samples:
            reacted_frac_table[sample] = (
                                                 self.count_table[sample] / self.sample_info[sample]['total_counts'] *
                                                 self.sample_info[sample]['quant_factor']
                                         )/reacted_frac_table.input_avg

        if inplace:
            self.reacted_frac_table = reacted_frac_table
        else:
            return reacted_frac_table

    def get_x_values(self, with_col_name=True):
        """Return x values corresponding to each column
        Args:

            with_col_name (bool): return a dict instead of a list if True

        """
        if with_col_name:
            return {
                sample: float(self.sample_info[sample]['x_value']) for sample in self.reacted_frac_table.columns
            }
        else:
            return [float(self.sample_info[sample]['x_value']) for sample in self.reacted_frac_table.columns]

    def filter_seq(self, seq_to_keep, inplace=True):
        """Filter sequence in dataset
        In current version, only ``count_table`` and ``reacted_frac_table`` will change if applicable
        Other meta info will keep the same

        Args:
            seq_to_keep (list of str): sequences to keep
            inplace: change in place if True

        Returns: Return a new SequenceSet object with only ``seq_to_keep`` if inplace is False

        """
        if inplace:
            self.count_table = self.count_table.loc[seq_to_keep]
            if hasattr(self, 'reacted_frac_table'):
                col_x_values = self.reacted_frac_table.col_x_values
                input_avg_type = self.reacted_frac_table.input_avg_type
                self.reacted_frac_table = self.reacted_frac_table.loc[seq_to_keep]
                self.reacted_frac_table.input_avg_type = input_avg_type
                self.reacted_frac_table.col_x_values = col_x_values
        else:
            import copy
            sequence_set_copy = copy.deepcopy(self)
            sequence_set_copy.count_table = self.count_table.loc[seq_to_keep]
            if hasattr(self, 'reacted_frac_table'):
                col_x_values = self.reacted_frac_table.col_x_values
                input_avg_type = self.reacted_frac_table.input_avg_type
                sequence_set_copy.reacted_frac_table = self.reacted_frac_table.loc[seq_to_keep]
                sequence_set_copy.reacted_frac_table.input_avg_type = input_avg_type
                sequence_set_copy.reacted_frac_table.col_x_values = col_x_values
            return sequence_set_copy



