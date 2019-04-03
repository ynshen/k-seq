"""
This module contains the methods for data input and output
"""

class SequencingSample:

    """
    This class defines and describe the experimental samples sequenced in k-seq experiments
    """

    def __init__(self):
        pass

    def read_count_file(self, file_root, sample_name, name_pattern=None):
        """
        read single count file to object SequencingSample
        :param file_root: root directory of the sample folder
        :param sample_name: the name (without directory) of the sample
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
        """
        self.dirc = '{}/{}'.format(file_root, sample_name)
        with open(self.dirc, 'r') as file:
            self.unique_seqs = int([elem for elem in next(file).strip().split()][-1])
            self.total_counts = int([elem for elem in next(file).strip().split()][-1])
            next(file)
            self.sequences = {}
            for line in file:
                seq = line.strip().split()
                self.sequences[seq[0]] = int(seq[1])
        if name_pattern:
            metadata = extract_sample_metadata(sample_name=sample_name, name_pattern=name_pattern)
            self.name = metadata.pop('name', None)
            self.metadata = metadata
            if 'input' in self.name or 'Input' in self.name:
                self.sample_type = 'input'
            else:
                self.sample_type = 'reacted'
        else:
            self.name = sample_name
            if 'input' in self.name or 'Input' in self.name:
                self.sample_type = 'input'
            else:
                self.sample_type = 'reacted'


class SequenceSet:
    def __init__(self):
        pass


def extract_sample_metadata(sample_name, name_pattern):
    import re

    def divide_string(string):
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

    def extract_domain_name(domain):
        digit_ix = max(domain.find(',digit'), domain.find(', digit'), domain.find(',d'), domain.find(', '))
        if digit_ix > 0:
            return (domain[:digit_ix], True)
        elif domain != '':
            return (domain, False)
        else:
            return None

    metadata = {}       # dict to save extracted values
    # Anchor the position of brackets and curly braces in name_pattern
    brackets = [name_pattern.find('['), name_pattern.find(']')]
    metadata['name'] = sample_name[brackets[0]:len(sample_name) + brackets[1] - len(name_pattern) + 1]
    curly_braces = [(0, brackets[0])] + \
                   list(zip([instance.start() for instance in re.finditer(string=name_pattern, pattern='{')],
                           [instance.start() for instance in re.finditer(string=name_pattern, pattern='}')])) + \
                   [(brackets[1], brackets[1]), (len(name_pattern) - 1, len(name_pattern) - 1)]
    sample_start = 0
    brace_ix = 1
    while brace_ix < len(curly_braces) and curly_braces[brace_ix][0] != curly_braces[brace_ix][1]:
        # get prefix
        prefix = name_pattern[curly_braces[brace_ix - 1][1] + 1: curly_braces[brace_ix][0]]
        domain_list = [extract_domain_name(name_pattern[curly_braces[brace_ix][0] + 1: curly_braces[brace_ix][1]])]
        # anchor the end of compound braces, e.g. {}{}
        while curly_braces[brace_ix][1] + 1 == curly_braces[brace_ix + 1][0]:
            brace_ix += 1
            if curly_braces[brace_ix][0] != curly_braces[brace_ix][1]:
                domain_list.append(
                    extract_domain_name(name_pattern[curly_braces[brace_ix][0] + 1: curly_braces[brace_ix][1]]))
        # get postfix
        postfix = name_pattern[curly_braces[brace_ix][1] + 1: curly_braces[brace_ix + 1][0]]
        # get region to extract
        start_ix = sample_name.find(prefix, sample_start) + len(prefix)
        sample_start = start_ix
        end_ix = sample_name.find(postfix, sample_start)
        sample_start = end_ix
        if len(domain_list) > 1:
            domain_values = divide_string(sample_name[start_ix:end_ix])
        else:
            domain_values = [sample_name[start_ix:end_ix]]
        for ix, domain in enumerate(domain_list):
            if domain[1]:
                metadata[domain[0]] = int(domain_values[ix])
            else:
                metadata[domain[0]] = domain_values[ix]
        brace_ix += 1
    return metadata


def get_file_list(file_root, pattern=None):
    """
    list all files under the given file root
    :param file_root: root directory
    :param pattern: optional, file name pattern to identify count files
    :return: a list of file names
    """
    import glob
    if not pattern:
        pattern = ''
    sample_list = [file_name[file_name.rfind('/')+1:]
                   for file_name in glob.glob("{}/*{}*".format(file_root, pattern))
                   if not '@' in file_name]
    sample_list.sort()
    return sample_list


def load_count_files(file_root, pattern=None, name_pattern=None, sort_fn=None, black_list=[]):
    """
    load all count files under file_root if comply with pattern. A list of SequencingSample will return, each includes
        self.dirc: full directory to the count file
        self.name: sample_name or indicated by name_pattern
        self.unqiue_seqs: number of unique sequences reported in count file
        self.total_counts: number of total counts reported in count file
        self.sequences: a dictionary of {seq: count} reported in count file
        self.sample_type: type of sample, either be 'input' or 'reacted'
    :param file_root: root directory
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
    :return:
    """
    sample_list = get_file_list(file_root=file_root, pattern=pattern)
    sample_set = []
    for sample_name in sample_list:
        if sample_name not in black_list:
            sample = SequencingSample()
            sample.read_count_file(file_root=file_root, sample_name=sample_name, name_pattern=name_pattern)
            sample_set.append(sample)
    if sort_fn:
        sample_set.sort(key = sort_fn)
    return sample_set


def convert_samples_to_sequences(sample_set, note=None):
    # find valid sequence set
    input_seq_set = []
    for sample in sample_set:
        if sample.sample_type == 'input':
            input_seq_set += list(sample.sequences.keys())
    input_seq_set = set(input_seq_set)
    reacted_seq_set = []
    for sample in sample_set:
        if sample.sample_type == 'reacted':
            reacted_seq_set += list(sample.sequences.keys())
    reacted_seq_set = set(reacted_seq_set)
    valid_set = input_seq_set && reacted_seq_set
    sequence_set = SequenceSet()
    sequence_set.input_seq_num = len(input_seq_set)
    sequence_set.reacted_seq_num = len(reacted_seq_set)
    sequence_set.valid_seq_num = len(valid_set)

    # preserve sample info
    for sample in sample_set:
        sequence_set.sample_info[sample.name] = {
            'unique_seqs': sample.unique_seqs,
            'total_counts': sample.total_counts,
            'sample_type': sample.sample_type,
            'passing_rate': get_passing_rate(sample, valid_set),
            'quant_factor': sample.quant_factor
        }


