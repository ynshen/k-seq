from yuning_util.dev_mode import DevMode
dev_mode = DevMode('k-seq')
dev_mode.on()

from k_seq.model import kinetic

from pytest import approx, raises
import numpy as np
import pandas as pd

# TODO: PoolModel can combine kinetic model and count model


def test_byo_reacted_frac_correct_value():
    c = [-1, 50e-6]
    k = 50
    A = 0.5
    results = kinetic.BYOModel.reacted_frac(c=c, k=k, A=A)
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
    results = kinetic.BYOModel.reacted_frac(c=c, k=k, A=A)
    np.testing.assert_array_almost_equal(results, np.array([[1, 0.051085207],
                                                            [1, 0.096951017]]))


def test_byo_reacted_frac_can_disable_broadcast():
    c = [-1, 50e-6]
    k = [50, 100]
    A = [0.5, 0.5]
    results = kinetic.BYOModel.reacted_frac(c=c, k=k, A=A, broadcast=False)
    np.testing.assert_array_almost_equal(results, np.array([[1, 0.051085207],
                                                            [1, 0.096951017]]))

