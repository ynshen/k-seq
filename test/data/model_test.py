from k_seq.model import kinetic, count, pool

import numpy as np

# TODO: PoolModel can combine kinetic model and count model


def test_poolModel_can_assign_param_and_have_correct_total_seq():
    # hard to validate each counts as it is randomly sampled
    pool_model = pool.PoolModel(
        count_model=count.multinomial,
        kinetic_model=kinetic.BYOModel.composition_first_order,
        p0=[0.1, 0.2, 0.4, 0.3],
        k=[10, 20, 10, 50],
        A=[0.8, 0.4, 0.1, 0.2]
    )
    assert np.sum(pool_model.predict(c=50e-6, N=int(10e6))[1]) == int(10e6)


def test_byo_reacted_frac_correct_value():
    c = [-1, 50e-6]
    k = 50
    A = 0.5
    results = kinetic.BYOModel.reacted_frac(broadcast=False)(c=c, k=k, A=A)
    np.testing.assert_array_almost_equal(results, np.array([1, 0.051085207]))


def test_byo_composition_correct_value():
    c = [-1, 50e-6]
    k = 50
    A = 0.5
    results = kinetic.BYOModel.composition_first_order(p0=[1], c=c, k=k, A=A)
    np.testing.assert_array_almost_equal(results, np.array([1, 0.051085207]))


def test_byo_reacted_frac_can_broadcast():
    c = [-1, 50e-6]
    k = [50, 100]
    A = 0.5
    results = kinetic.BYOModel.reacted_frac(broadcast=True)(c=c, k=k, A=A)
    np.testing.assert_array_almost_equal(results, np.array([[1, 0.051085207],
                                                            [1, 0.096951017]]))


def test_byo_reacted_frac_can_disable_broadcast():
    c = [-1, 50e-6]
    k = [50, 100]
    A = [0.5, 0.5]
    results = kinetic.BYOModel.reacted_frac(broadcast=False)(c=c, k=k, A=A)
    np.testing.assert_array_almost_equal(results, np.array([[1, 0.051085207],
                                                            [1, 0.096951017]]))


def test_multinomial_correct_return():
    N = 50
    p = (0.1, 0.5, 0.3, 0.1)
    assert count.multinomial(N=N, p=p).sum() == 50

