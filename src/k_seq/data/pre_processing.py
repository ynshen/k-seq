"""This module contains methods for data preprocessing from count files to ``SequenceSet`` for fitting
"""

import numpy as np
import pandas as pd


def load_count_files(file_root, x_values, sample_list=None, pattern=None, name_pattern=None, sort_fn=None, black_list=[], silent=True):
    """
    load all count files under file_root if comply with pattern. A list of SequencingSample will return, each includes
        self.file_dirc: full directory to the count file
        self.name: sample_name or indicated by name_pattern
        self.unqiue_seqs: number of unique sequences reported in count file
        self.total_counts: number of total counts reported in count file
        self.sequences: a dictionary of {seq: count} reported in count file
        self.sample_type: type of sample, either be 'input' or 'reacted'
    :param file_root: root directory
    :param x_values: string or a list of floats. if string, will use it as domain name to extract x_value for each sample
                     if a list of floats, it should have same length and order as the sample list
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
    if sample_list is None:
        sample_list = get_file_list(file_root=file_root, pattern=pattern)
        print("NOTICE: no sample_list is given, samples will extract automaticall from file_root.")
        if type(x_values) != str:
            raise Exception("No sample_list is given, please indicate domain name instead of list of real values to extract x_values")
    sample_set = []
    if type(x_values) == str:
        x_values = [x_values for _ in sample_list]
    for sample_ix,sample_name in enumerate(sample_list):
        if sample_name not in black_list:
            sample = SequencingSample(file_root=file_root,
                                      sample_name=sample_name,
                                      x_value=x_values[sample_ix],
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

