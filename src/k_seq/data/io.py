"""
This module contains the methods for data input and output
"""

class SequencingSample:

    """
    This class defines and describe the experimental samples sequenced in k-seq experiments
    """

    def __init__(self):
        pass


def extract_sample_metadata(sample_name, name_pattern):

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
    metadata['name'] = sample[brackets[0]:brackets[1] - len(name_pattern) + 1]
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



def load_count_files(file_root, ):
    from os import listdir
    from os.path import isfile, join

    sampleList = [f for f in listdir(file_root) if isfile(join(countFileRoot, f))]
    sort_fn = lambda s: int(s.split('_')[1][1:])
    sampleList.sort(key=sort_fn)
    return countFileRoot, sampleList

def get_sample_list(file_root, pattern=None):
    import glob

    if not pattern:
        pattern = ''
    sample_list = [file_name[file_name.rfind('/')+1:]
                   for file_name in glob.glob("{}/*{}*".format(file_root, pattern)) if not '@' in file_name]
    return sample_list
