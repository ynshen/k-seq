import os
from k_seq.data.seq_table import SeqTable
import numpy as np


def test_count_file_can_load():
    count_file_dir = os.getenv('COUNT_FILE_DIR')
    pattern_filter = '_counts.'
    name_template = 'd-[{byo}{exp_rep}]_S{smpl}_counts.txt'

    seq_table = SeqTable.from_count_files(
        count_files=count_file_dir,
        pattern_filter=pattern_filter,
        name_template=name_template,
        dry_run=False,
        sort_by='smpl',
        x_values=np.concatenate((
            np.repeat([1250, 250, 50, 10, 2], repeats=3) * 1e-6,
            np.array([np.nan])), axis=0),
        x_unit='M',
        input_sample_name=['R0']
    )
