"""Private
Pipeline for generating datasets for k-seq project
Available datasets:
  - BYO doped pool: byo-doped
  - BYO selection pool: byo-selected
"""

import os
from yutility import logging


def load_dataset(dataset, from_count_file=False, **kwargs):
    """Load default dataset
    Available dataset:
      - BYO-doped: 'byo-doped'
      - BYO-selected: 'byo-selected'
      - BFO: not implemented
    """
    if dataset.lower() in ['byo_doped', 'byo-doped', 'byo-variant', 'byo_variant']:
        from .byo_doped import load_byo_doped
        return load_byo_doped(from_count_file=from_count_file, **kwargs)
    elif dataset.lower() in ['byo-doped-test', 'byo_doped_test']:
        from .byo_doped import load_byo_doped
        return load_byo_doped(from_count_file=from_count_file,
                              count_file_path=os.getenv('BYO_DOPED_COUNT_FILE_TEST'),
                              **kwargs)
    elif dataset.lower() in ['byo_selected', 'byo-selected', 'selected', 'byo-selection', 'byo-enriched', 'byo_enriched']:
        from .byo_selected import load_byo_selected
        return load_byo_selected(from_count_file=from_count_file, **kwargs)
    else:
        logging.error(f'Dataset {dataset} is not implemented', error_type=NotImplementedError)
