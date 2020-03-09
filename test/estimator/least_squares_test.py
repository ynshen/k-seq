import numpy as np
import pandas as pd


def test_SingleFitter_can_run_all():
    from k_seq.estimator import SingleFitter

    def model(x, k, b):
        return k * x + b

    def kb(param):
        return param[0] * param[1]

    def bs_stats(record):
        return record.sum(axis=0)

    x_data = np.logspace(0, 3, 20)
    y_data = model(x_data, 4, 3)

    single_fitter = SingleFitter(x_data=x_data, y_data=y_data, model=model, metrics={'kb': kb}, rnd_seed=23,
                                 bootstrap_num=10, bs_record_num=5, bs_method='data', bs_stats={'num': bs_stats},
                                 conv_reps=10, conv_init_range=[(0, 1), (-5, 5)], conv_stats={'num': bs_stats})

    single_fitter.fit(point_estimate=True, bootstrap=True, convergence_test=True)

    assert isinstance(single_fitter.results.convergence.records, pd.DataFrame)
    assert isinstance(single_fitter.results.uncertainty.summary, pd.Series)
    assert isinstance(single_fitter.summary(), pd.Series)


def kA(param):
    return param[0] * param[1]


def spearman(records):
    from scipy import stats
    return stats.spearmanr(records['k'], records['A']).correlation


def pearson(records):
    from scipy import stats
    return stats.pearsonr(records['k'], records['A'])[0]


def normality_shapiro_test(records):
    from scipy import stats
    return stats.shapiro(records['kA'])[0]


def get_BatchFitter(large_dataset=False, result_path=None):
    from k_seq.estimator import BatchFitter
    from k_seq.data import datasets
    from k_seq.model import kinetic

    byo_doped = datasets.load_dataset('byo-doped-test', from_count_file=True)

    return BatchFitter(
        y_dataframe=byo_doped.table.reacted_frac_qpcr,
        x_data=byo_doped.x_values,
        model=kinetic.BYOModel.reacted_frac(broadcast=False),
        bounds=[(0, 0), (np.inf, 1)],
        metrics={'kA': kA},
        bootstrap_num=5,
        bs_record_num=3,
        bs_stats={'spearman': spearman,
                  'normality_shapiro_test': normality_shapiro_test},
        conv_reps=5,
        conv_init_range=[(0, 1), (0, 1)],
        conv_stats={'spearman': spearman, 'pearson': pearson},
        large_dataset=large_dataset,
        result_path=result_path
    )


def test_BatchFitter_can_run_small_dataset():
    import os

    batch_fitter = get_BatchFitter(large_dataset=False)
    output_dir = os.getenv('KSEQ_TEST_OUTPUT')
    batch_fitter.fit(bootstrap=True, convergence_test=True, parallel_cores=8,
                     point_estimate=True)

    batch_fitter.results.to_pickle(f'{output_dir}/small_dataset.pkl')
    del batch_fitter
    batch_fitter = get_BatchFitter(large_dataset=False, result_path=f'{output_dir}/small_dataset.pkl')
    assert isinstance(batch_fitter.results.bs_record('CTTCTTCAAACAATCGGTCTG'), pd.DataFrame)
    os.remove(f'{output_dir}/small_dataset.pkl')


def test_BatchFitter_can_run_large_dataset():
    import os

    batch_fitter = get_BatchFitter(large_dataset=True)
    output_dir = os.getenv('KSEQ_TEST_OUTPUT')
    batch_fitter.fit(bootstrap=True, convergence_test=True, overwrite=True, parallel_cores=8,
                     point_estimate=True, stream_to=f'{output_dir}/large_dataset/')
    batch_fitter.results.to_json(f'{output_dir}/large_dataset/')

    del batch_fitter
    batch_fitter = get_BatchFitter(large_dataset=True, result_path=f'{output_dir}/large_dataset/')
    assert isinstance(batch_fitter.results.bs_record('CTTCTTCAAACAATCGGTCTG'), pd.DataFrame)
    import shutil
    shutil.rmtree(f'{output_dir}/large_dataset/')
