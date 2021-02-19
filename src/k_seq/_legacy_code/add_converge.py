"""
Archived code to run convergence test
NOT compatible with version 0.3.0+
"""

import util
import numpy as np
import sys
sys.path.append('/home/yuning/Work/k-seq/source/')
import k_seq_eval
import multiprocessing as mp
import time


def add_config(seq):
    validSampleList = [
        [1, 2, 3, 4, 5, 6],
        [7, 8, 10, 11, 12],
        [13, 14, 15],
        [19, 20, 21, 23]
    ]
    return [sum([1 for rep in repSet if ~np.isnan(seq['reactedFrac'][rep])]) for repSet in validSampleList]


def load_fittingRes(dirc):
    fittingRes = util.load_pickle(dirc)
    for seq in fittingRes:
        seq['config'] = add_config(seq)
    return fittingRes

def work_fn(res):
    valid = ~np.isnan(res['reactedFrac'])
    x = np.array([0]
                 + [0.00025 for i in range(6)]
                 + [0.00005 for i in range(6)]
                 + [0.00001 for i in range(6)]
                 + [0.000002 for i in range(6)])
    convergence = k_seq_eval.fitting_convergence_test(x=x[valid], y=res['reactedFrac'][valid], rep=20, retMod='Full')
    res['convergTest'] = convergence
    return res

def main():
    timeInit = time.time()
    fittingRes = load_fittingRes('/mnt/storage/projects/ribozyme_predict/k_seq/repeat_res/fittingRes_byo_abeData_fix.pkl')
    #fittingRes = fittingRes[:1000]
    pool = mp.Pool(processes=6)
    fittingRes = pool.map(work_fn, fittingRes)
    util.dump_pickle(data=fittingRes,
                     dirc='/mnt/storage/projects/ribozyme_predict/k_seq/repeat_res/fittingRes_byo_abeData_fix_add_converge.pkl',
                     log='Fitting results of Abe\'s data, the convergence test is added',
                     overwrite=True)
    timeEnd = time.time()
    print('Process finished in %i s' % (timeEnd - timeInit))


if __name__ == '__main__':
    main()
