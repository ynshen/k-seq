"""Test code for log module"""

from yuning_util.dev_mode import DevMode
dev_mode = DevMode('k-seq')
dev_mode.on()

from k_seq.utility import log
import pandas as pd


def test_Logger_can_log():
    logger = log.Logger()
    assert isinstance(logger.log, pd.DataFrame)

    logger.info('Some message')
    assert logger.log.iloc[-1]['message'] == 'Some message'
