
def get_file_list(file_root, pattern=None, file_list=None, black_list=None, full_path=True):
    """list all files under the given file root(s) follows the pattern if given, folders are not included

    Args:
        file_root (`str` of list of `str`): root directory or a list of root directory

        pattern (`str`): optional, include all the files under directories if None

        file_list (list of `str`): optional, only includes the files with names in the file_list

        black_list (list of `str`): optional, file name include substring in black list will be excluded

        full_path (`bool`): if return the full path or only name of the file, by default, if file_root is one string,
          only file name will be returned; if file_root contains multiple strings, full path will be returned

    Returns:

        list of `str` or `path.Path`: list of file names/full directory
    """

    from pathlib import Path

    if pattern is None:
        pattern = '*'
    else:
        pattern = '*{}*'.format(pattern)

    if black_list is None:
        black_list = []

    if isinstance(file_root, str):
        if full_path is None:
            full_path = False
        files = [file for file in Path(file_root).glob(pattern) if file.name not in black_list]

    elif isinstance(file_root, list):
        if full_path is None:
            full_path = True
        files = []
        for root_path in file_root:
            files += [file for file in Path(root_path).glob(pattern) if file.name not in black_list]
    else:
        raise TypeError('file_root should be string or list of string')

    if file_list is not None:
        files = [file for file in files if str(file.name) in file_list]

    if full_path:
        return files
    else:
        return [file.name for file in files]


def extract_metadata(target, pattern):
    """Function to extract metadata info from target given pattern
    todo: update method to account for missing metadata situation. e.g. matching -A- to -{d1}{d2}-
    Args:

        target (`str`): string to extract info from, e.g. sample file name

        pattern (`str`): string indicate the method to extract metadata. Use ``[...]`` to include the region of sample_name,
          use ``{domain_name[, int/float]}`` to indicate region of domain to extract as metadata, including
          ``[,int/float]`` will convert the domain value to float in applicable, otherwise, string

    Return:

        dict: dictionary of all metadata extracted from domains indicated in ``pattern``

    Example:
        Example on metadata extraction from pattern:

        >>> extract_metadata(
                sample_name = "R4B-1250A_S16_counts.txt"
                pattern = "R4[{exp_rep}-{concentration, float}{seq_rep}_S{id, int}]_counts.txt"
            )
        metadata = {
            'name': 'B-1250A_S16',
            'exp_rep': 'B',
            'concentration': 1250.0,
            'seq_rep': 'A',
            'id': 16
            }

    """
    import numpy as np

    def extract_info_from_braces(target, pattern):
        """
        Iterative algorithm to extract metadata info from target and pattern
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
                    return (domain[0], np.int)
                elif 'float' in domain[1] or 'f' in domain[1]:
                    return (domain[0], np.float32)
                else:
                    return (domain[0], str)
            else:
                return (domain, str)

        def get_domains(pattern):
            """
            inspect pattern and extract domain(s) from it
            multiple domains could be expected because of {}{} structure
            """

            domains = pattern[1:-1].split('}{')
            return [parse_domain(domain) for domain in domains]

        def extract_info(target, prefix, postfix):
            """
            extract substring in target are flanked by pre-fix and post-fix
            prefix: no need to find, always the first len(prefix) character of target
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

        # anchor braces in pattern
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
        # anchor info domain in target from prefix and postfix
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
    # Anchor the position of brackets and curly braces in name_pattern
    brackets = [pattern.find('['), pattern.find(']')]
    metadata = extract_info_from_braces(target=target,
                                        pattern=pattern[:brackets[0]] + '{name}' + pattern[brackets[1] + 1:])
    metadata.update(extract_info_from_braces(target=metadata['name'],
                                             pattern=pattern[brackets[0] + 1: brackets[1]]))

    return metadata


def read_pickle(path):
    """Read pickled object form path"""
    import pickle

    with open(path, 'rb') as handle:
        return pickle.load(handle)


def dump_pickle(obj, path):
    import pickle

    with open(path, 'wb') as handle:
        pickle.dump(obj, handle)


def read_json(path):
    import json

    with open(path, 'r') as handle:
        return json.load(handle)


def dump_json(obj, path, indent=4):
    import json

    with open(path, 'w') as handle:
        json.dump(obj, handle, indent=indent)


def check_dir(path):
    """Check if a path exists, create if not"""
    from pathlib import Path
    if Path(path).exists():
        return True
    else:
        Path(path).mkdir(parents=True)
        return False
