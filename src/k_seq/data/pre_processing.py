"""
This module contains the methods for data input and output
"""

import numpy as np
import pandas as pd

class SequencingSample:

    """
    This class defines and describe the experimental samples sequenced in k-seq experiments
    """

    def __init__(self):
        pass

    def read_count_file(self, file_root, sample_name, name_pattern=None):
        """
        read single count file to object SequencingSample
        :param file_root: root directory of the sample folder
        :param sample_name: the name (without directory) of the sample
        :param name_pattern: optional. pattern to extract metadata. pattern rules: [...] to include the region of
               sample_name, {domain_name[, digit]} to indicate region of domain to extract as metadata, including
               [,digit] will convert the domain value to number, otherwise, string
               e.g. R4B-1250A_S16_counts.txt with pattern = "R4[{exp_rep}-{concentration, digit}{seq_rep}_S{id, digit}_counts.txt"
               will return SequencingSample.metadata = {
                                    'exp_rep': 'B',
                                    'concentration': 1250.0,
                                    'seq_rep': 'A',
                                    'id': 16.0
                                 }
        """
        self.file_dirc = '{}/{}'.format(file_root, sample_name)
        with open(self.file_dirc, 'r') as file:
            self.unique_seqs = int([elem for elem in next(file).strip().split()][-1])
            self.total_counts = int([elem for elem in next(file).strip().split()][-1])
            next(file)
            self.sequences = {}
            for line in file:
                seq = line.strip().split()
                self.sequences[seq[0]] = int(seq[1])
        if name_pattern:
            metadata = extract_sample_metadata(sample_name=sample_name, name_pattern=name_pattern)
            self.name = metadata.pop('name', None)
            self.metadata = metadata
            if 'input' in self.name or 'Input' in self.name:
                self.sample_type = 'input'
            else:
                self.sample_type = 'reacted'
        else:
            self.name = sample_name
            if 'input' in self.name or 'Input' in self.name:
                self.sample_type = 'input'
            else:
                self.sample_type = 'reacted'

    def survey_spike_in(self, spike_in='AAAAACAAAAACAAAAACAAA', max_dist_to_survey=10):
        """
        This method will survey the spike in sequences in each sample
        :param stdSeq:
        :return:
        """
        import Levenshtein

        self.spike_in_counts = np.array([0 for _ in range(max_dist_to_survey + 1)])
        for seq in self.sequences.keys():
            dist = Levenshtein.distance(spike_in, seq)
            if dist <= max_dist_to_survey:
                self.spike_in_counts[dist] += self.sequences[seq]
        self.spike_in = spike_in

    def get_quant_factor(self, max_dist, spike_in_amount):
        """
        Add quant_factor and quant_factor_max_dist attributes to SequencingSample
        :param max_dist:
        :param spike_in_amount:
        :return:
        """
        self.quant_factor = spike_in_amount * self.total_counts / self.spike_in_counts[max_dist]
        self.quant_factor_max_dist = max_dist
        self.spike_in_amount = spike_in_amount



class SequenceSet:
    def __init__(self):
        pass


def extract_sample_metadata(sample_name, name_pattern):
    import re

    def divide_string(string):
        def letter_label(letter):
            if letter.isdigit():
                return 0
            elif letter.isupper():
                return 1
            else:
                return 2

        label = [letter_label(letter) for letter in string]

        split_ix = [-1] + [ix for ix in range(len(string) - 1) if label[ix] != label[ix + 1]] + [len(string)]
        return [string[split_ix[i] + 1: split_ix[i + 1] + 1] for i in range(len(split_ix) - 1)]

    def extract_domain_name(domain):
        digit_ix = max(domain.find(',digit'), domain.find(', digit'), domain.find(',d'), domain.find(', '))
        if digit_ix > 0:
            return (domain[:digit_ix], True)
        elif domain != '':
            return (domain, False)
        else:
            return None

    metadata = {}       # dict to save extracted values
    # Anchor the position of brackets and curly braces in name_pattern
    brackets = [name_pattern.find('['), name_pattern.find(']')]
    metadata['name'] = sample_name[brackets[0]:len(sample_name) + brackets[1] - len(name_pattern) + 1]
    curly_braces = [(0, brackets[0])] + \
                   list(zip([instance.start() for instance in re.finditer(string=name_pattern, pattern='{')],
                           [instance.start() for instance in re.finditer(string=name_pattern, pattern='}')])) + \
                   [(brackets[1], brackets[1]), (len(name_pattern) - 1, len(name_pattern) - 1)]
    sample_start = 0
    brace_ix = 1
    while brace_ix < len(curly_braces) and curly_braces[brace_ix][0] != curly_braces[brace_ix][1]:
        # get prefix
        prefix = name_pattern[curly_braces[brace_ix - 1][1] + 1: curly_braces[brace_ix][0]]
        domain_list = [extract_domain_name(name_pattern[curly_braces[brace_ix][0] + 1: curly_braces[brace_ix][1]])]
        # anchor the end of compound braces, e.g. {}{}
        while curly_braces[brace_ix][1] + 1 == curly_braces[brace_ix + 1][0]:
            brace_ix += 1
            if curly_braces[brace_ix][0] != curly_braces[brace_ix][1]:
                domain_list.append(
                    extract_domain_name(name_pattern[curly_braces[brace_ix][0] + 1: curly_braces[brace_ix][1]]))
        # get postfix
        postfix = name_pattern[curly_braces[brace_ix][1] + 1: curly_braces[brace_ix + 1][0]]
        # get region to extract
        start_ix = sample_name.find(prefix, sample_start) + len(prefix)
        sample_start = start_ix
        end_ix = sample_name.find(postfix, sample_start)
        sample_start = end_ix
        if len(domain_list) > 1:
            domain_values = divide_string(sample_name[start_ix:end_ix])
        else:
            domain_values = [sample_name[start_ix:end_ix]]
        for ix, domain in enumerate(domain_list):
            if domain[1]:
                metadata[domain[0]] = int(domain_values[ix])
            else:
                metadata[domain[0]] = domain_values[ix]
        brace_ix += 1
    return metadata


def get_file_list(file_root, pattern=None):
    """
    list all files under the given file root
    :param file_root: root directory
    :param pattern: optional, file name pattern to identify count files
    :return: a list of file names
    """
    import glob
    if not pattern:
        pattern = ''
    sample_list = [file_name[file_name.rfind('/')+1:]
                   for file_name in glob.glob("{}/*{}*".format(file_root, pattern))
                   if not '@' in file_name]
    sample_list.sort()
    return sample_list


def load_count_files(file_root, pattern=None, name_pattern=None, sort_fn=None, black_list=[]):
    """
    load all count files under file_root if comply with pattern. A list of SequencingSample will return, each includes
        self.file_dirc: full directory to the count file
        self.name: sample_name or indicated by name_pattern
        self.unqiue_seqs: number of unique sequences reported in count file
        self.total_counts: number of total counts reported in count file
        self.sequences: a dictionary of {seq: count} reported in count file
        self.sample_type: type of sample, either be 'input' or 'reacted'
    :param file_root: root directory
    :param pattern: optional, file name pattern to identify count files
    :param name_pattern: optional. pattern to extract metadata. pattern rules: [...] to include the region of
                         sample_name, {domain_name[, digit]} to indicate region of domain to extract as metadata, including
                         [,digit] will convert the domain value to number, otherwise, string
                         e.g. R4B-1250A_S16_counts.txt with pattern = "R4[{exp_rep}-{concentration, digit}{seq_rep}_S{id, digit}_counts.txt"
                         will return SequencingSample.metadata = {
                                              'exp_rep': 'B',
                                              'concentration': 1250.0,
                                              'seq_rep': 'A',
                                              'id': 16.0
                                           }
    :param sort_fn: optional, callable to sort sample order
    :param black_list: name of sample files that will be excluded in loading
    :return:
    """
    sample_list = get_file_list(file_root=file_root, pattern=pattern)
    sample_set = []
    for sample_name in sample_list:
        if sample_name not in black_list:
            sample = SequencingSample()
            sample.read_count_file(file_root=file_root, sample_name=sample_name, name_pattern=name_pattern)
            sample_set.append(sample)
    if sort_fn:
        sample_set.sort(key = sort_fn)
    return sample_set


def get_quant_factors(sample_set, spike_in='AAAAACAAAAACAAAAACAAA', max_dist=2, max_dist_to_survey=10,
                      spike_in_amounts=None, manual_quant_factor=None):
    if manual_quant_factor:
        for sample_ix, sample in enumerate(sample_set):
            sample.quant_factor = manual_quant_fasctor[sample_ix]
    else:
        for sample_ix, sample in enumerate(sample_set):
            sample.survey_spike_in(spike_in = spike_in, max_dist_to_survey=max_dist_to_survey)
            sample.get_quant_factor(max_dist, spike_in_amounts[sample_ix])
    return sample_set


def convert_samples_to_sequences(sample_set, remove_spike_in=True, note=None):
    # TODO: use object.__dict__ instead of copying
    # TODO: reorganize the structure of object data storage
    """
    Convert a list of SequencingSample objects to a SequenceSet object. A typical SequenceSet object includes:
        self.input_seq_num: number of unique sequences in all "input" samples
        self.reacted_seq_num number of unique sequences in all "reacted" samples
        self.valid_seq_num: number of valid unqiue sequences that detected in at least one "input" sample and one
                            "reacted" sample
        self.sample_info: a list of dictionaries, containing the information from original samples
        self.count_table: a pandas.DataFrame object of valid sequences and their original counts in samples
        self.valid_seq_remove_spike_in: Boolean. If True, sequences considered as spike-in will not include in counting
        self.note: Optional. Addtional notes regarding to the dataset
    :param sample_set: a list of SequencingSample objects to convert
    :param remove_spike_in: Boolean. See above
    :param note: Optional. See above
    :return: A SequenceSet object
    """

    import Levenshtein

    # find valid sequence set
    input_seq_set = []
    reacted_seq_set = []
    if remove_spike_in:
        for sample in sample_set:
            if sample.sample_type == 'input':
                input_seq_set += list([seq for seq in sample.sequences.keys()
                                       if Levenshtein.distance(seq, sample.spike_in) > sample.quant_factor_max_dist])
        for sample in sample_set:
            if sample.sample_type == 'reacted':
                reacted_seq_set += list([seq for seq in sample.sequences.keys()
                                         if Levenshtein.distance(seq, sample.spike_in) > sample.quant_factor_max_dist])
    else:
        for sample in sample_set:
            if sample.sample_type == 'input':
                input_seq_set += list(sample.sequences.keys())
        input_seq_set = set(input_seq_set)
        reacted_seq_set = []
        for sample in sample_set:
            if sample.sample_type == 'reacted':
                reacted_seq_set += list(sample.sequences.keys())

    input_seq_set = set(input_seq_set)
    reacted_seq_set = set(reacted_seq_set)
    valid_set = input_seq_set & reacted_seq_set
    sequence_set = SequenceSet()
    sequence_set.note = note
    sequence_set.input_seq_num = len(input_seq_set)
    sequence_set.reacted_seq_num = len(reacted_seq_set)
    sequence_set.valid_seq_num = len(valid_set)
    sequence_set.valid_seq_remove_spike_in = remove_spike_in

    # preserve sample info
    sequence_set.sample_info = {}
    for sample in sample_set:
        sequence_set.sample_info[sample.name] = {
            'unique_seqs': sample.unique_seqs,
            'total_counts': sample.total_counts,
            'sample_type': sample.sample_type,
            'valid_seqs': np.sum([1 for seq in sample.sequences.keys() if seq in valid_set]),
            'quant_factor': sample.quant_factor,
        }
        if hasattr(sample, 'quant_factor_max_dist'):
            sequence_set.sample_info[sample.name]['quant_factor_max_dist'] = sample.quant_factor_max_dist
        if hasattr(sample, 'spike_in_amount'):
            sequence_set.sample_info[sample.name]['spike_in_amount'] = sample.spike_in_amount
        if hasattr(sample, 'metadata'):
            sequence_set.sample_info[sample.name]['metadata'] = sample.metadata
    # create valid sequence table
    sequence_set.count_table = pd.DataFrame(index = list(valid_set), columns=[sample.name for sample in sample_set])
    for seq in valid_set:
        for sample in sample_set:
            if seq in sample.sequences.keys():
                sequence_set.count_table.loc[seq, sample.name] = sample.sequences[seq]

    return sequence_set

def survey_seqs_info(sequence_set):
    sequence_set.seq_info = pd.DataFrame(index = sequence_set.count_table.index)
    input_samples = [sample[0] for sample in sequence_set.sample_info.items() if sample[1]['sample_type'] == 'input']
    reacted_samples = [sample[0] for sample in sequence_set.sample_info.items() if sample[1]['sample_type'] == 'reacted']
    sequence_set.seq_info['occur_in_inputs'] = pd.Series(
        np.sum(sequence_set.count_table.loc[:, input_samples] > 0, axis=1),
        index=sequence_set.count_table.index
    )
    sequence_set.seq_info['occur_in_reacteds'] = pd.Series(
        np.sum(sequence_set.count_table.loc[:, reacted_samples] > 0, axis=1),
        index=sequence_set.count_table.index
    )
    sequence_set.seq_info['total_counts_in_inputs'] = pd.Series(
        np.sum(sequence_set.count_table.loc[:, input_samples], axis=1),
        index=sequence_set.count_table.index
    )
    sequence_set.seq_info['total_counts_in_reacteds'] = pd.Series(
        np.sum(sequence_set.count_table.loc[:, reacted_samples], axis=1),
        index=sequence_set.count_table.index
    )
    return sequence_set


def get_reacted_frac(sequence_set, input_average='median', black_list=None, inplace=False):
    if not black_list:
        black_list = []
    input_samples = [sample[0] for sample in sequence_set.sample_info.items()
                     if sample[0] not in black_list and sample[1]['sample_type'] == 'input']
    reacted_samples = [sample[0] for sample in sequence_set.sample_info.items()
                       if sample[0] not in black_list and sample[1]['sample_type'] == 'reacted']
    reacted_frac_table = pd.DataFrame(index=sequence_set.count_table.index)
    avg_method = 'input_{}'.format(input_average)
    if input_average == 'median':
        input_amount_avg = np.nanmedian(np.array([
            list(sequence_set.count_table[sample] / sequence_set.sample_info[sample]['total_counts'] *
            sequence_set.sample_info[sample]['quant_factor'])
            for sample in input_samples
        ]), axis=0)
    elif input_average == 'mean':
        input_amount_avg = np.nanmean(np.array([
            list(sequence_set.count_table[sample] / sequence_set.sample_info[sample]['total_counts'] *
                 sequence_set.sample_info[sample]['quant_factor'])
            for sample in input_samples
        ]), axis=0)
    else:
        raise Exception("Error: input_average should be 'median' or 'mean'")
    reacted_frac_table[avg_method] = input_amount_avg
    for sample in reacted_samples:
        reacted_frac_table[sample] = (
            sequence_set.count_table[sample] / sequence_set.sample_info[sample]['total_counts'] *
            sequence_set.sample_info[sample]['quant_factor']
        )/reacted_frac_table[avg_method]
    for sample in input_samples:
        reacted_frac_table[sample] = (
            sequence_set.count_table[sample] / sequence_set.sample_info[sample]['total_counts'] *
            sequence_set.sample_info[sample]['quant_factor']
        )/reacted_frac_table[avg_method]
    if inplace:
        sequence_set.reacted_frac_table = reacted_frac_table
    else:
        return reacted_frac_table