from doc_helper import DocHelper
import yutility
import os
import multiprocessing as mp
from tqdm import tqdm

logging = yutility.log.logging
info = logging.info

allowed_units = ['g', 'mg', 'ug', 'ng',
                 'mol', 'mmol', 'umol', 'fmol']


def mp_job(fn, itr, n_proc=os.cpu_count(), chunksize=None, use_fork=False, **kwargs):
    """Run jobs in the iterator in parallel"""
    from functools import partial
    if use_fork:
        info('Using fork mode - this could be unsafe for subprocesses')
        mp.set_start_method('fork', force=True)

    total = len(itr)
    if kwargs != {}:
        fn = partial(fn, **kwargs)

    if chunksize is None:
        chunksize = max(total // n_proc // 2, 1)

    with mp.Pool(n_proc) as pool:
        res = list(tqdm(pool.imap(fn, itr, chunksize=chunksize), total=total))
    return res