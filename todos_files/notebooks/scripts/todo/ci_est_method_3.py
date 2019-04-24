import k_seq.confidence_estimation
import numpy as np
import util

simuSet = util.load_pickle('/mnt/storage/projects/ribozyme_predict/data/k_seq/fit_simu/simuSet_2_full.pkl')
for sample in simuSet:
    var, std = k_seq.confidence_estimation.method_3(sample)
    sample['estVar'] = var
    sample['estStd'] = std
util.dump_pickle(data = simuSet,
                 dirc = '/mnt/storage/projects/ribozyme_predict/data/k_seq/fit_simu/simuSet_2_full_method_3_res.pkl',
                 log = 'Estimation results from method 3 on simuSet_2_full',
                 overwrite = True)