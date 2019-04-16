"""
This module contains the methods for data input and output
"""

import numpy as np
import pandas as pd
from . import io


class SequencingSample:

    """
    This class defines and describe the experimental samples sequenced in k-seq experiments
    """

    def __init__(self, file_root, sample_name, silent=True, name_pattern=None):
        """
        initialize a SequencingSample instance by reading single count file
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
        import datetime

        self.metadata = {}
        if file_root[-1] != '/':
            file_root += '/'
        self.metadata['file_dirc'] = '{}{}'.format(file_root, sample_name)
        self.unique_seq, self.total_counts, self.sequences = io.read_count_file(self.metadata['file_dirc'])

        if name_pattern:
            metadata = extract_sample_metadata(sample_name=sample_name, name_pattern=name_pattern)
            self.name = metadata.pop('name', None)
            self.metadata.update(metadata)
        else:
            self.name = sample_name

        if 'input' in self.name or 'Input' in self.name:
            self.sample_type = 'input'
        else:
            self.sample_type = 'reacted'

        self.metadata['timestamp'] = str(datetime.datetime.now())
        if not silent:
            print("Sample {} imported.".format(self.name))

    def survey_spike_in(self, spike_in, max_dist_to_survey=10, silent=True):
        """
        This method will survey the number of spike-in sequences in the sample, with edit distance to the center
        spike-in sequence
        Following attributes will be added to the instance:
        - spike_in: dict, {
            spike_in_counts: list of int with length max_dist_to_survey + 1, number of total counts with distance i to
                             the center spike-in sequence
            spike_in: string, spike_in sequence
          }
        :param spike_in: string, the sequence of spike-in, consider as the center sequence
        :param max_dist_to_survey: int, the maximum distance to survey
        :return: None
        """
        import Levenshtein

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
        """
        Add quant_factor and quant_factor_max_dist attributes to SequencingSample
        quant_factor here is defined as spike_in_amount/total_counts/np.sum(spike_in_counts[:max_dist + 1])
        :param max_dist:
        :param spike_in_amount:
        :return:
        """

        self.quant_factor = spike_in_amount * self.total_counts / np.sum(self.spike_in_counts[:max_dist + 1])
        self.spike_in['quant_factor_max_dist'] = max_dist
        self.spike_in['spike_in_amount'] = spike_in_amount

        if not silent:
            print("Calculate quant-factor for sample {}. Done.".format(self.name))


def extract_sample_metadata(sample_name, name_pattern):
    """
    Auxiliary function to extract sample information from sample_name, provided name_pattern
    :param sample_name: string, sample name
    :param name_pattern: pattern to extract metadata.
                         pattern rules:
                             [...] to include the region of sample_name,
                             {domain_name[, digit]} to indicate region of domain to extract as metadata, including
                             [,digit] will convert the domain value to number, otherwise, string
                         e.g. R4B-1250A_S16_counts.txt
                              with pattern = "R4[{exp_rep}-{concentration, digit}{seq_rep}_S{id, digit}_counts.txt"
                              will return SequencingSample.metadata = {
                                              'exp_rep': 'B',
                                              'concentration': 1250.0,
                                              'seq_rep': 'A',
                                              'id': 16.0
                                           }
    :return: metadata
    """
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
            return domain[:digit_ix], True
        elif domain != '':
            return domain, False
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
                try:
                    metadata[domain[0]] = float(domain_values[ix])
                except:
                    metadata[domain[0]] = domain_values[ix]
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


def load_count_files(file_root, pattern=None, name_pattern=None, sort_fn=None, black_list=[], silent=True):
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
    :param slient: boolean, false: print calculation progress to the screen
    :return: sample_set: a list of SequencingSample class
    """
    sample_list = get_file_list(file_root=file_root, pattern=pattern)
    sample_set = []
    for sample_name in sample_list:
        if sample_name not in black_list:
            sample = SequencingSample(file_root=file_root,
                                      sample_name=sample_name,
                                      name_pattern=name_pattern,
                                      silent=silent)
            sample_set.append(sample)
    if sort_fn:
        sample_set.sort(key=sort_fn)
    return sample_set


def get_quant_factors(sample_set, spike_in='AAAAACAAAAACAAAAACAAA', max_dist=2, max_dist_to_survey=10,
                      spike_in_amounts=None, manual_quant_factor=None, silent=True):
    """
    Assign/calculate quant_factor for SequencingSet instances in sample_set
    :param sample_set:
    :param spike_in:
    :param max_dist:
    :param max_dist_to_survey:
    :param spike_in_amounts:
    :param manual_quant_factor:
    :param silent:
    :return:
    """
    if manual_quant_factor:
        for sample_ix, sample in enumerate(sample_set):
            sample.quant_factor = manual_quant_fasctor[sample_ix]
    else:
        for sample_ix, sample in enumerate(sample_set):
            sample.survey_spike_in(spike_in = spike_in, max_dist_to_survey=max_dist_to_survey, silent=silent)
            sample.get_quant_factor(spike_in_amount = spike_in_amounts[sample_ix], max_dist=max_dist)

    return sample_set


class SequenceSet:

    def __init__(self, sample_set, remove_spike_in=True, note=None):
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
            sample_info_dict = sample.__dict__
            sequences = sample_info_dict.pop('sequences', None)
            self.sample_info[sample.name] = {
                'valid_seqs_num': np.sum([1 for seq in sequences.keys() if seq in valid_set]),
                'valid_seqs_counts': np.sum([seq[1] for seq in sequences.items() if seq[0] in valid_set])
            }
            self.sample_info[sample.name].update(sample_info_dict)

        # create valid sequence table
        self.count_table = pd.DataFrame(index = list(valid_set), columns=[sample.name for sample in sample_set])
        for seq in valid_set:
            for sample in sample_set:
                if seq in sample.sequences.keys():
                    self.count_table.loc[seq, sample.name] = sample.sequences[seq]

    def survey_seqs_info(self):
        self.seq_info = pd.DataFrame(index = self.count_table.index)
        input_samples = [sample[0] for sample in self.sample_info.items() if sample[1]['sample_type'] == 'input']
        reacted_samples = [sample[0] for sample in self.sample_info.items() if sample[1]['sample_type'] == 'reacted']
        self.seq_info['occur_in_inputs'] = pd.Series(
            np.sum(self.count_table.loc[:, input_samples] > 0, axis=1),
            index=self.count_table.index
        )
        self.seq_info['occur_in_reacteds'] = pd.Series(
            np.sum(self.count_table.loc[:, reacted_samples] > 0, axis=1),
            index=self.count_table.index
        )
        self.seq_info['total_counts_in_inputs'] = pd.Series(
            np.sum(self.count_table.loc[:, input_samples], axis=1),
            index=self.count_table.index
        )
        self.seq_info['total_counts_in_reacteds'] = pd.Series(
            np.sum(self.count_table.loc[:, reacted_samples], axis=1),
            index=self.count_table.index
        )
        return self


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


def get_replicates(sequence_set, key_domain):
    from itertools import groupby

    sample_type = [(sample[0], sample[1]['metadata'][key_domain]) for sample in sequence_set.sample_info.items()]
    sample_type.sort(key=lambda x: x[1])
    groups = {}
    for key, group in groupby(sample_type, key=lambda x: x[1]):
        groups[key] = [x[0] for x in group]
    return groups