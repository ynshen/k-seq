import sys
import numpy as np
if np.sum([dirc.find('k-seq') >= 0 for dirc in sys.path]) == 0:
    sys.path = ['/home/yuning/Work/k-seq/src/'] + sys.path

from src import k_seq as pre_processing, k_seq as fitting


def main():
    # read counts files sample_set
    sample_set = pre_processing.load_count_files(
        file_root='/mnt/storage/projects/k-seq/input/bfo_counts/counts',
        pattern='_counts.txt',
        x_values='byo',
        name_pattern='[R4{select_rep}-{byo, digit}{sequence_rep}_S{sample_id, digit}]_counts.txt',
        sort_fn=lambda sample: sample.metadata['sample_id']
    )
    print('Sequencing samples imported')

    # indicate spike-in amount to calculate quantification factors
    spike_in_amounts = []
    for i in range(4):
        spike_in_amounts += [4130, 1240, 826, 413, 207, 82.6, 41.3]

    sample_set = pre_processing.get_quant_factors(
        sample_set=sample_set,
        max_dist=2,
        max_dist_to_survey=10,
        spike_in='AAAAACAAAAACAAAAACAAA',
        spike_in_amounts=spike_in_amounts
    )
    print('Quantification factors calculated')

    # TODO: save sample_set

    # extract valid sequences and convert data into sequence_set
    sequence_set = pre_processing.SequenceSet(sample_set=sample_set, remove_spike_in=True,
                                              note='Valid sequences from all samples, spike-in counts are removed.')
    print('Valid sequences are extracted')
    zero_samples = [sample_name for sample_name in sequence_set.sample_info.keys() if '-0' in sample_name]
    sequence_set.get_reascted_frac(
        input_average='median',
        black_list=zero_samples,
        inplace=True
    )
    print('Reacted fraction calculated')

    # estimator
    print('Start estimator valide sequences...')
    fitting.fitting_sequence_set(sequence_set=sequence_set_test, inplace=True, parallel_threads=6)

    # TODO: save estimator results


