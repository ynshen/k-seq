"""Parse, convert, characterize count files generated from Chen lab's customized scripts

TODO: add Sam and Celia's code reference here
"""

from ..utility.log import logging
from .seq_data import _doc
from ..utility.file_tools import _name_template_example
import numpy as np
import pandas as pd


@_doc.compose(f"""Create a ``SeqTable`` instance from a folder of count files

Args:
    count_files (str): root directory to search for count files
    file_list (list of str): optional, only includes the count files with names in the file_list
    pattern_filter (str): optional, filter file names based on this pattern, wildcards ``*/?`` are allowed
    black_list (list of str): optional, file names included in black_list will be excluded
    name_template (str): naming convention to extract metadata. Use ``[...]`` to include the region of sample_name,
        use ``{{domain_name[, int/float]}}`` to indicate region of domain to extract as metadata, including
        ``int`` or ``float`` will convert the domain value to int/float in applicable, otherwise, string
    sort_by (str): sort the order of samples based on given domain
    dry_run (bool): only return the parsed count file names and metadata without actual reading in data
<<x_values, x_unit, input_sample_name, sample_metadata, note>>

{_name_template_example}

""")
def load_Seqtable_from_count_files(
    count_files, file_list=None, pattern_filter=None, black_list=None, name_template=None, sort_by=None,
    x_values=None, x_unit=None, input_sample_name=None, sample_metadata=None, note=None,
    dry_run=False
):

    from ..utility.file_tools import get_file_list, extract_metadata

    # parse file metadata
    file_list = get_file_list(file_root=count_files, file_list=file_list,
                              pattern=pattern_filter, black_list=black_list, full_path=True)
    if name_template is None:
        samples = {file.name: {'file_path': str(file), 'name': file.name} for file in file_list}
    else:
        samples = {}
        for file in file_list:
            f_meta = extract_metadata(name=file.name, template=name_template)
            samples[f_meta['name']] = {**f_meta, **{'file_path': str(file)}}
    if sample_metadata is not None:
        for file_name, f_meta in sample_metadata.items():
            samples[file_name].udpate(f_meta)

    # sort file order if applicable
    sample_names = list(samples.keys())
    if sort_by is not None:
        if isinstance(sort_by, str):
            def sort_fn(sample_name):
                return samples[sample_name].get(sort_by, np.nan)
        elif callable(sort_by):
            sort_fn = sort_by
        else:
            logging.error('Unknown sort_by format', error_type=TypeError)
        sample_names = sorted(sample_names, key=sort_fn)

    if dry_run:
        # return a list of samples without importing
        return pd.DataFrame(samples)[sample_names].transpose()

    data_mtx = {sample: read_count_file(file_path=samples[sample]['file_path'], as_dict=True)[2]
                for sample in sample_names}
    data_mtx = pd.DataFrame.from_dict(data_mtx).fillna(0, inplace=False).astype(pd.SparseDtype(dtype='int'))
    if input_sample_name is not None:
        grouper = {'input': [name for name in sample_names if name in input_sample_name],
                   'reacted': [name for name in sample_names if name not in input_sample_name]}
    else:
        grouper = None

    from .seq_data import SeqData

    seq_table = SeqData(data_mtx, data_unit='count', grouper=grouper, sample_metadata=sample_metadata,
                        x_values=x_values, x_unit=x_unit, note=note)

    return seq_table


_count_file_format = """Count file format:
::
    number of unique sequences = 2825
    total number of molecules = 29348173

    AAAAAAAACACCACACA               2636463
    AATATTACATCATCTATC              86763
    ...
"""


def read_count_file(file_path, as_dict=False, number_only=False):
    """Read a single count file generated from Chen lab's customized scripts
    
    {}

    Args:
        file_path (str): full directory to the count file
        as_dict (bool): return a dictionary instead of a `pd.DataFrame`
        number_only (bool): only return number of unique seqs and total counts if True

    Returns:
        unique_seqs (int): number of unique sequences in the count file
        total_counts (int): number of total reads in the count file
        sequence_counts (pd.DataFrame): with ``sequence`` as index and ``counts`` as the first column
    """.format(_count_file_format)

    with open(file_path, 'r') as file:
        unique_seqs = int([elem for elem in next(file).strip().split()][-1])
        total_counts = int([elem for elem in next(file).strip().split()][-1])
        if number_only:
            sequence_counts = None
            as_dict = True
        else:
            next(file)
            sequence_counts = {}
            for line in file:
                seq = line.strip().split()
                sequence_counts[seq[0]] = int(seq[1])

    if as_dict:
        return unique_seqs, total_counts, sequence_counts
    else:
        return unique_seqs, total_counts, pd.DataFrame.from_dict(sequence_counts, orient='index', columns=['counts'])
