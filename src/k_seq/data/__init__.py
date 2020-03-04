"""This sub-package includes the modules for data handling, including:

    * ``pre_processing``: core module in data pre-processing from count file to *SequenceSeq* for estimator
    * ``io``: module contains utility function for read, write and convert different file formats
    * ``analysis``: module contains functions for extra analysis for sequencing samples or reads to sample investigation
      and sample pipeline quality control
    * ``simu``: module contains codes to generate simulated data used in analysis in paper (TODO: add paper citation)
"""


from . import visualization, simu
import landscape

