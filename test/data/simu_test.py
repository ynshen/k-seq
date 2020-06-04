"""Test code for simulation"""

from k_seq.data import simu
from k_seq.data.seq_data import SeqData
from pytest import approx
import numpy as np
import pandas as pd


def check_returns_correct_df(returned, size):
    assert isinstance(returned, pd.DataFrame)
    assert returned.shape[0] == size
    assert returned.p0.sum() == approx(1)


def test_sample_from_iid_dist_takes_list():
    size = 3
    list = [1, 2, 3]
    check_returns_correct_df(
        simu.PoolParamGenerator.sample_from_iid_dist(uniq_seq_num=size, p0=list),
        size=size
    )


def test_sample_from_iid_dist_takes_unmatch_list():
    size = 3
    list = [1, 2, 3, 4, 5, 6]
    check_returns_correct_df(
        simu.PoolParamGenerator.sample_from_iid_dist(uniq_seq_num=size, p0=list),
        size=size
    )


def fake_gen():
    while True:
        yield np.random.random()


def test_sample_from_iid_dist_takes_generator():
    generator = fake_gen()
    size = 3
    case = simu.PoolParamGenerator.sample_from_iid_dist(uniq_seq_num=size, p0=generator, param1=generator)
    check_returns_correct_df(case, size=size)
    assert case.shape[1] == 2


def test_sample_from_iid_dist_takes_callable_for_list():

    def fake_callable_return_list_w_size(size):
        return np.arange(size)

    size = 3
    callable_list = fake_callable_return_list_w_size
    case = simu.PoolParamGenerator.sample_from_iid_dist(uniq_seq_num=size, p0=callable_list, param1=callable_list)
    check_returns_correct_df(case, size=size)
    assert case.shape[1] == 2


def test_sample_from_iid_dist_takes_callable_for_generator():

    def fake_callable_return_generator_no_size():
        return fake_gen()

    size = 3
    callable_gen = fake_callable_return_generator_no_size
    case = simu.PoolParamGenerator.sample_from_iid_dist(uniq_seq_num=size, p0=callable_gen, param1=callable_gen)
    check_returns_correct_df(case, size=size)
    assert case.shape[1] == 2


def test_sample_from_dataframe_returns_df():
    size = 3
    df = pd.DataFrame(data={'A': [1, 2, 3, 4, 5], 'k': [1, 2, 3, 4, 5]})
    check_returns_correct_df(simu.PoolParamGenerator.sample_from_dataframe(df=df, uniq_seq_num=size, replace=True),
                             size=size)


def test_sample_from_dataframe():
    size = 3
    df = pd.DataFrame(data={'A': [1, 2, 3, 4, 5], 'k': [1, 2, 3, 4, 5]})
    weights = [0.3, 0.1, 0.9, 0.1, 0.3]
    check_returns_correct_df(simu.PoolParamGenerator.sample_from_dataframe(df=df, uniq_seq_num=size, weights=weights),
                             size=size)


def test_simulate_counts_return_correct():
    x, Y, dna_amount, param_table, seq_table = simu.simulate_counts(
        uniq_seq_num=10,
        x_values=[-1, 2e-6, 50e-6],
        total_reads=1000,
        p0_generator=[0.1, 0.5, 0.4, 0.6, 0.2],
        reps=3,
        k=[10, 5, 30],
        A=[0.8, 0.9]
    )
    assert x.shape == (2, 9)
    assert Y.shape == (10, 9)
    assert all([val == 1000 for val in Y.sum(axis=0)])
    assert param_table.shape == (10, 3)
    assert isinstance(seq_table, SeqData)


def test_simulate_w_byo_doped_condition_from_param_dist_returns_correct_shape():
    x, Y, dna_amount, truth, seq_table = simu.simulate_w_byo_doped_condition_from_param_dist(
        uniq_seq_num=20, depth=40, p0_loc=1, p0_scale=0.1, k_95=(0.1, 100),
        total_dna_error_rate=0.1, save_to=None, plot_dist=False
    )
    assert x.shape == (2, 16)
    assert Y.shape == (20, 16)
    assert Y.iloc[:, 0].sum() == 2400
    assert all([val == 800 for val in Y.iloc[:, 1:].sum(axis=0)])
    assert truth.shape == (20, 4)
    assert isinstance(seq_table, SeqData)


# def test_simulate_w_byo_doped_condition_from_exp_results_can_return():
    # TODO: Need to avoid load data from csv files
    # x, Y, dna_amount, truth, seq_table = simu.simulate_on_byo_doped_condition_from_exp_results(
    #     uniq_seq_num=20, depth=40, p0_loc=1, p0_scale=0.1, k_95=(0.1, 100),
    #     total_dna_error_rate=0.1, save_to=None, plot_dist=False
    # )
    # assert x.shape == (2, 16)
    # assert Y.shape == (20, 16)
    # assert Y.iloc[:, 0].sum() == 2400
    # assert all([val == 800 for val in Y.iloc[:, 1:].sum(axis=0)])
    # assert truth.shape == (20, 4)
    # assert isinstance(seq_table, SeqData)

