"""Modules for data handling, including:

    * ``preprocessing``: core module in data pre-processing from count file to *SequenceSeq* for estimator
    * ``io``: module contains utility function for read, write and convert different file formats
    * ``analysis``: module contains functions for extra analysis for sequencing samples or reads to sample investigation
      and sample pipeline quality control
    * ``simu``: module contains codes to generate simulated data used in analysis in the paper
"""

from . import visualization, simu
import landscape


def axis_mapper(axis):
    """Map the name of data table axis to axis number
    axis 0: seq, sequence, sequences
    axis 1: sample, observation, obs
    """
    if isinstance(axis, int):
        if axis in [0, 1]:
            return axis
    elif isinstance(axis, str):
        if axis.lower() in ['seq', 'sequence', 'sequences']:
            return 0
        elif axis.lower() in ['sample', 'samples', 'observation', 'obs']:
            return 1
        else:
            pass
    else:
        pass
    raise ValueError(f"Unknown axis: {axis}. Allowed: 0 (seq, sequence, sequences), 1 (sample, observation, obs)")
