"""Test code for simulation"""
from yuning_util.dev_mode import DevMode
dev_mode = DevMode('k-seq')
dev_mode.on()

from k_seq.data import simu
from pytest import approx, raises
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



# TODO: simulate_counts returns x, Y, parameters, and a SeqTable

def test_simulate_counts_return_correct():
    simu.simulate_counts(uniq_seq_num=10,
                         x_values=[-1, 2e-6, 50e-6],
                         total_reads=1000,
                         p0=[0.1, 0.5, 0.4, 0.6, 0.2],
                         reps=3,
                         k=[10, 5, 30],
                         A=[0.8, 0.9])
