import numpy as np
import pandas as pd


def test_SingleFitter_can_run_all():
    from k_seq.estimator.least_squares import SingleFitter

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
