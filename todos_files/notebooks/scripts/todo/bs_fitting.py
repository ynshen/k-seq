import util
from src import k_seq as k_seq
import time
import multiprocessing as mp
import numpy as np

def work_fn(seq):
    xdata = np.array([0.00025 for i in range(6)] +
                     [0.00005 for i in range(6)] +
                     [0.00001 for i in range(6)] +
                     [0.000002 for i in range(6)])
    seq = k_seq.fitting(seq, xdata, fitMtd='trf', ciEst=True)
    return seq

def main():
    timeInit = time.time()
    seqToFit = util.load_pickle('/mnt/storage/projects/ribozyme_predict/k_seq/seqToFit.pkl')
    # seqToFit = np.random.choice(seqToFit, uniq_seq_num=10, replace=False)
    pool = mp.Pool(processes=6)
    seqToFit = pool.map(work_fn, seqToFit)
    util.dump_pickle(data=seqToFit,
                     dirc='/mnt/storage/projects/ribozyme_predict/k_seq/fittingRes_abe.pkl',
                     log='Fitting results of Abe\'s data, including the CI 95 estimation using 500 bootstrap',
                     overwrite=True)
    timeEnd = time.time()
    print('Process finished in %i s' % (timeEnd - timeInit))

if __name__ == '__main__':
    main()
