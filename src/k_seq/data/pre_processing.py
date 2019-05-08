"""This module contains methods for data preprocessing from count files to ``SequenceSet`` for fitting
"""

import numpy as np
import pandas as pd
from . import SequencingSample, SequenceSet


def load_count_files(file_root, x_values, sample_list=None, pattern=None, name_pattern=None, sort_fn=None, black_list=[], silent=True):
    """load count files under a root folder into :func:`~SequencingSample` objects

    Args:
        file_root (str): directory to the root folder

        x_values (str or list of float): string or a list of floats. The time points or concentration points value for
          each sample. If string, the function will use it as domain name to extract x_value from file name; if a list
          of floats, it should have same length and order as sample file under file_root
          (Use :func:`~k_seq.utility.get_file_list' to examine the files automatically extracted)

        pattern (str): optional, file name pattern to identify count files. Only file with name strictly contains the pattern
          will be collected.

        name_pattern (str): optional. Pattern to extract metadata, see :func:`~k_seq.utility.extract_metadata'

        sort_fn (callable): optional. A callable to customize sample order

        sample_list (list of str): optional, if there is a given list of file name

        black_list (list of str): name of sample files that will be excluded in loading

        silent (boolean): don't print process if True

    Returns:
        list of :func:`~SequencingSample`
    """
    if sample_list is None:
        sample_list = get_file_list(file_root=file_root, pattern=pattern)
        if not silent:
            print("NOTICE: no sample_list is given, samples will extract automatically from file_root.")
        if type(x_values) != str:
            raise Exception("No sample_list is given, please indicate domain name instead of list of real values to extract x_values")
    sample_set = []
    if type(x_values) == str:
        x_values = [x_values for _ in sample_list]
    for sample_ix,sample_name in enumerate(sample_list):
        if sample_name not in black_list:
            sample = SequencingSample(file_dirc=file_root + sample_name,
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

