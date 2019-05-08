"""This sub-package includes the modules for data handling, including:

    * ``pre_processing``: core module in data pre-processing from count file to *SequenceSeq* for fitting
    * ``io``: module contains utility function for read, write and convert different file formats
    * ``analysis``: module contains functions for extra analysis for sequencing samples or reads to sample investigation
      and sample pipeline quality control
    * ``simu``: module contains codes to generate simulated data used in analysis in paper (TODO: add paper citation)
"""


class SequencingSample:
    """This class describes experimental samples sequenced in k-seq
    """

    def __init__(self, file_dirc, x_value, silent=True, name_pattern=None):
        """``SequencingSample`` instance store the data of each sequencing sample in k-seq experiment
        Initialize a ``SequencingSample`` instance by reading a count file

        Args:
            file_dirc (str): directory to the count file of sample

            x_value (float or str): corresponding x_value for the sample. If string, will automatically
                inspect the domain with the name

            silent (bool): optional. Print progress to std.out if True

            name_pattern (str): optional. pattern to extract metadata using :func:`~k_seq.utility.extract_metadata`.
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

        from . import io
        import datetime
        from .. import utility
        import numpy as np

        self.metadata = {}
        sample_name = file_dirc[file_dirc.rfind('/'):]
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
            print("Sample {} imported.".format(self.name))

    def survey_spike_in(self, spike_in, max_dist_to_survey=10, silent=True):
        """Survey spike-in counts in the sample.
        Count number of sequences at different edit distance to spike-in sequences (external standard for quantification)
            in the sample. Add attribute ``spike_in`` to the instance.

        Args:
            spike_in (str): the sequence of spike-in, as the center sequence

            max_dist_to_survey (int): the maximum distance to survey

            silent (bool): don't print progress if True

        Attributes:

            spike_in (dict):

                - spike_in_counts (list): length max_dist_to_survey + 1, number of total counts with different edit
                  distance to spike-in sequence

                - spike_in (str): center spike_in sequence

        """
        import Levenshtein
        from . import io
        import numpy as np

        self.spike_in = {}
        self.spike_in['spike_in_counts'] = np.array([0 for _ in range(max_dist_to_survey + 1)])
        self.spike_in['spike_in'] = spike_in
        for seq in self.sequences.keys():
            dist = Levenshtein.distance(spike_in, seq)
            if dist <= max_dist_to_survey:
                self.spike_in['spike_in_counts'][dist] += self.sequences[seq]
        if not silent:
            print("Survey spike-in counts for sample {}. Done.".format(self.name))

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
        from . import io
        import numpy as np

        self.quant_factor = spike_in_amount * self.total_counts / np.sum(self.spike_in['spike_in_counts'][:max_dist + 1])
        self.spike_in['quant_factor_max_dist'] = max_dist
        self.spike_in['spike_in_amount'] = spike_in_amount

        if not silent:
            print("Calculate quant-factor for sample {}. Done.".format(self.name))


class SequenceSet:
    """This class contains the dataset of valid sequences extracted and aligned from a list of ``SequencingSample``
    """

    def __init__(self, sample_set, remove_spike_in=True, note=None):
        """Initialize from a list of ``SequencingSample`` to a SequenceSet object.
        Find all valid sequences that occur at least once in any 'input' sample and once in any 'reacted' sample

        Args:

            sample_set (list of ``SequencingSample``): valid samples to convert to ``SequenceSet``

            remove_spike_in (bool): sequences considered as spike-in will be removed

            note (str): optional. Additional note to add to the dataset


        Attributes:

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
        from . import io
        import datetime
        import numpy as np

        # find valid sequence set
        input_seq_set = set()
        reacted_seq_set = set()

        if remove_spike_in:
            for sample in sample_set:
                if sample.sample_type == 'input':
                    input_seq_set.update([
                        seq for seq in sample.sequences.keys()
                        if Levenshtein.distance(seq, sample.spike_in['spike_in']) > sample.spike_in['quant_factor_max_dist']
                    ])
                elif sample.sample_type == 'reacted':
                    reacted_seq_set.update([
                        seq for seq in sample.sequences.keys()
                        if Levenshtein.distance(seq, sample.spike_in['spike_in']) > sample.spike_in['quant_factor_max_dist']
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

        from . import io
        import numpy as np

        if not black_list:
            black_list = []
        input_samples = [sample[0] for sample in self.sample_info.items()
                         if sample[0] not in black_list and sample[1]['sample_type'] == 'input']
        reacted_samples = [sample[0] for sample in self.sample_info.items()
                           if sample[0] not in black_list and sample[1]['sample_type'] == 'reacted']
        reacted_frac_table = pd.DataFrame(index=self.count_table.index, columns=reacted_samples)
        reacted_frac_table.input_avg_type = input_average
        reacted_frac_table.col_x_values = [self.sample_info[sample]['x_value'] for sample in reacted_frac_table.columns]

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
