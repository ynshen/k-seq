import os
from k_seq.data.seq_data import SeqData
import numpy as np


def test_count_file_can_load():
    count_file_dir = os.getenv('BYODOPED_COUNT_FILE_DIR')
    pattern_filter = '_counts.'
    name_template = 'd-[{byo}{exp_rep}]_S{smpl}_counts.txt'

    seq_table = SeqData.from_count_files(
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

    assert seq_table.table.original.shape[1] == 16
    assert hasattr(seq_table.grouper, 'input')


def test_count_file_can_load_from_multiple_source():
    seq_data = SeqData.from_count_files(
        count_files=[
            {
                'count_files': os.getenv('BYODOPED_COUNT_FILE_DIR'),
                'pattern_filter': '_counts.',
                'name_template': 'd-[{byo}{exp_rep}]_S{smpl}_counts.txt'
            },
            {
                'count_files': os.getenv('BFODOPED_COUNT_FILE_DIR'),
                'pattern_filter': '_top50.',
                'name_template': '[input_{input_id}]_counts_top50.txt'
            }
        ],
        dry_run=True,
        sort_by=None,
        input_sample_name=['R0']
    )
    assert seq_data.shape[0] == 22


def test_can_load_byo_doped():
    from k_seq.data import datasets
    byo_doped = datasets.load_dataset(
        'byo-doped', from_count_file=True,
        count_file_path=os.getenv('BYODOPED_COUNT_FILE_DIR'),
        doped_norm_path=os.getenv('BYODOPED_NORM_DIR'),
        pandaseq_joined=True
    )
