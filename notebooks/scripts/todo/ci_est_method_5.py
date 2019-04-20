import util
import time
import k_seq.confidence_estimation

if __name__ == '__main__':
    timeInit = time.time()
    simuSet = util.load_pickle('/mnt/storage/projects/ribozyme_predict/data/k_seq/fit_simu/simuSet_3_full.pkl')
    simuSet = k_seq.confidence_estimation.method_5_multi_main(simuSet)
    timeEnd = time.time()
    util.dump_pickle(simuSet,
                     '/mnt/storage/projects/ribozyme_predict/data/k_seq/fit_simu/simuSet_3_full_method_5_parallel_res.pkl',
                     log='Estimation on Simuset_3_full: bootstrap residues for 500 times, parallelized',
                     overwrite=True)
    print('Wall time: %.2f' %(timeEnd - timeInit))
