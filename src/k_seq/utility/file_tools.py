from .log import logging
from pathlib import Path


def get_file_list(file_root, pattern=None, file_list=None, black_list=None, full_path=True):
    """Return files under the given `file root` match the `template` if applicable, folders are not included

    Args:
        file_root (str of list of str): root directory/directories to search
        pattern (str): optional, include all the files under directories if None
        file_list (list of str): optional, only includes the files with names in the file_list if exists
        black_list (list of str): optional, file names included in black_list will be excluded
        full_path (bool): if return the full path or only name of the file, by default, if file_root is one string,
          only file name will be returned; if file_root contains multiple strings, full path will be returned

    Returns:
        list of str (file names) or path.Path (full directory)
    """

    if pattern is None:
        pattern = '*'
    else:
        pattern = '*{}*'.format(pattern)

    if black_list is None:
        black_list = []

    if isinstance(file_root, (str, Path)):
        files = [file for file in Path(file_root).glob(pattern) if file.name not in black_list]
    elif isinstance(file_root, list):
        files = []
        for root_path in file_root:
            files += [file for file in Path(root_path).glob(pattern) if file.name not in black_list]
    else:
        logging.error('count_files should be a string or list of string', error_type=TypeError)

    if file_list is not None:
        files = [file for file in files if str(file.name) in file_list]

    if full_path:
        return files
    else:
        return [file.name for file in files]


_name_template_example = """
Example:

    Example on metadata extraction from pattern:
    >>> metadata = extract_metadata(
            sample_name = "R4B-1250A_S16_counts.txt"
            template = "R4[{exp_rep}-{concentration, float}{seq_rep}_S{id, int}]_counts.txt"
        )

    >>> metadata
    {
        'name': 'B-1250A_S16',
        'exp_rep': 'B',
        'concentration': 1250.0,
        'seq_rep': 'A',
        'id': 16
    }

Notice: two back-to-back domain can only be parsed if one of them is numeric and one of them is alphabetic, and missing
    value will raise error

    Valid: matching '-A1-' to '-{{sample}}{{replicate, int}}-' gives {{ 'sample': 'A', 'replicate': 1}}
    Not valid: matching '-A-' to '-{{sample}}{{replicate, int}}-' will cause error
               matching '-AA-' to '-{{sample}}{{replicate}}-' will cause error
"""


_extract_metadata_doc = f"""Function to extract metadata info from a string (name, e.g. file name) given a template
    indicating the position of each metadata domain.
    
Args:

    name (str): string to extract info, e.g. sample file name
    template (str): naming convention to extract metadata. Use ``[...]`` to include the region of sample_name,
      use ``{{domain_name[, int/float]}}`` to indicate region of domain to extract as metadata, including
      ``int`` or ``float`` will convert the domain value to int/float in applicable, otherwise, string

Return:

    dict: dictionary of all metadata extracted from domains indicated in ``pattern``

{_name_template_example} 
    
"""


def extract_metadata(name, template):

    import numpy as np

    def extract_info_from_braces(target, pattern):
        """
        Iterative algorithm to extract metadata info from name and template
        """

        def stop(string, ix):
            """stop conditions when finding the rightest index of current domain(s)"""
            if ix == len(string) - 1:
                return True
            if string[ix] == '}' and string[ix + 1] != '{':
                return True
            else:
                return False

        def parse_domain(domain):
            """parse domain name and type"""
            if ',' in domain:
                domain = domain.split(',')
                if 'int' in domain[1] or 'i' in domain[1]:
                    return domain[0], np.int
                elif 'float' in domain[1] or 'f' in domain[1]:
                    return domain[0], np.float32
                else:
                    return domain[0], str
            else:
                return (domain, str)

        def get_domains(pattern):
            """
            inspect template and extract domain(s) from it
            multiple domains could be expected because of {}{} structure
            """

            domains = pattern[1:-1].split('}{')
            return [parse_domain(domain) for domain in domains]

        def extract_info(target, prefix, postfix):
            """
            extract substring in name are flanked by pre-fix and post-fix
            prefix: no need to find, always the first len(prefix) character of name
            postfix: the first occurrence of postfix substring after prefix, if not ''
            """
            if postfix == '':
                return target[len(prefix):]
            else:
                return target[len(prefix):target.find(postfix, len(prefix))]

        def divide_string(string):
            """
            split a string into consecutive chunks of digits, letters and upper letters
            """

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

        # anchor braces in template
        brace_left = pattern.find('{')
        if brace_left == -1:
            return {}
        brace_right = brace_left
        while not stop(pattern, brace_right):
            brace_right += 1
        domains = get_domains(pattern[brace_left:brace_right + 1])
        # find prefix and postfix
        prefix = pattern[:brace_left]
        postfix = pattern[brace_right + 1:pattern.find('{', brace_right + 1)
        if pattern.find('{', brace_right + 1) != -1 else len(pattern)]
        # anchor info domain in name from prefix and postfix
        info = extract_info(target, prefix, postfix)
        if len(domains) > 1:
            info = divide_string(info)
            if len(info) > len(domains):
                info[len(domains) - 1] = ''.join(info[len(domains) - 1:])
        else:
            info = [info]
        info_list = dict()
        for ix, domain in enumerate(domains):
            try:
                info_list[domain[0]] = domain[1](info[ix])
            except:
                info_list[domain[0]] = str(info[ix])
        # iteratively calculate the leftover substring
        if postfix != '':
            info_list.update(extract_info_from_braces(target=target[target.find(postfix, len(prefix)):],
                                                      pattern=pattern[brace_right + 1:]))
        return info_list

    metadata = {}  # dict to save extracted values
    # Anchor the position of brackets and curly braces in name_template
    brackets = [template.find('['), template.find(']')]
    metadata = extract_info_from_braces(target=name,
                                        pattern=template[:brackets[0]] + '{name}' + template[brackets[1] + 1:])
    metadata.update(extract_info_from_braces(target=metadata['name'],
                                             pattern=template[brackets[0] + 1: brackets[1]]))

    return metadata


extract_metadata.__doc__ = _extract_metadata_doc


def read_pickle(path):
    """Read pickled object form path"""
    import pickle

    with open(path, 'rb') as handle:
        return pickle.load(handle)


def dump_pickle(obj, path):
    """Save object as picked file"""
    import pickle

    with open(path, 'wb') as handle:
        pickle.dump(obj, handle)


def read_json(path):
    """Read json file"""
    import json

    with open(path, 'r') as handle:
        return json.load(handle)


def dump_json(obj, path=None, indent=2):
    """Convert object to a JSON file or JSON string"""
    import json

    if path:
        with open(path, 'w') as handle:
            json.dump(obj, handle, indent=indent)
    else:
        return json.dumps(obj)


def read_table_files(file_path, col_name=None, header=1):
    """Read common table files
    - .xls or .xlsx: first sheet will be read with first row as header
    - .csv: read the csv files with first row as header, separator is ','
    - .tsv: read the tsv files with first row as header, separator is '/t'
    """
    from pathlib import Path
    import pandas as pd

    file_path = Path(file_path)
    if file_path.suffix in ['xls', 'xlsx']:
        df = pd.read_excel(io=file_path, sheet_name=0, header=header)
    elif file_path.suffix in ['csv']:
        df = pd.read_csv(file_path, header=header)
    elif file_path.suffix in ['tsv']:
        df = pd.read_csv(file_path, header=header, sep='/t')
    else:
        logging.error('File type not identified', error_type=TypeError)

    return df[col_name]


def check_dir(path):
    """Check if a path exists, create if not"""
    from pathlib import Path
    if Path(path).exists():
        return True
    else:
        Path(path).mkdir(parents=True)
        return False


def table_object_to_dataframe(obj, table_name=None):
    """Convert object (`file path`, `SeqData`) to `pd.DataFrame`
    """
    from pathlib import Path, PosixPath
    import pandas as pd
    from ..data.seq_data import SeqData

    if isinstance(obj, (str, Path, PosixPath)):
        if Path(obj).is_file():
            try:
                obj = read_pickle(obj)
            except:
                raise TypeError(f'{obj} is not pickled object')
        else:
            raise FileNotFoundError(f'{obj} is not a valid file')
    if isinstance(obj, pd.DataFrame):
        return obj
    elif isinstance(obj, SeqData):
        if table_name is None:
            try:
                return obj.table
            except AttributeError:
                raise AttributeError('Please indicate the table name')
        else:
            try:
                return getattr(obj, table_name)
            except AttributeError:
                raise AttributeError(f'{table_name} is not found in the SeqData object')
    else:
        raise TypeError('SeqTable should be a `pd.DataFrame` or `SeqData`')
