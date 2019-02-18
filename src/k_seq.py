'''
This module contain methods for k-seq calculation
'''
__author__ = "Yuning Shen"
__email__ = "yuningshen@ucsb.edu"


def get_q_factor(dirc, std, amount, maxDist=1):
    """
    ** BE AWARE: the q_factor has been changed to its original's reciprocal for an easier definition **
    calculate quantification factor from a k-seq sample
    :param dirc: k-seq sample count file dirc
    :param std: spike-in sequence
    :param amount: amount of spike-in sequence
    :param maxDist: maximum distance of edit distanced counted as standard
    :return: (qFactor, stdCount, total)
    """
    import Levenshtein

    stdCount = 0
    with open(dirc) as file:
        next(file)
        total = int([elem for elem in next(file).strip().split()][-1])
        next(file)
        for line in file:
            seq = line.strip().split()
            dist = Levenshtein.distance(std, seq[0])
            if dist <= maxDist:
                stdCount += int(seq[1])

    return (total*amount/stdCount, stdCount, total)

def get_seqToFit(initDirc, sampleDircList, validSampleList):
    """
    This function generate a list of sequences that will pass to fitting
    Three steps: 1) create candidate list; 2) align each sample to candidate list; 3) select sequences with interests
    :param initDirc: directory to the count file of initial pool
    :param sampleDircList: FULL list of directories to all k-seq samples
    :param validSampleList: a list of ids of valid samples that will be considered in this function, PlEASE NOTE: 1) the
    id MUST coordinate with the order of samples in sampleDircList, starting from 1; 2) the list should format in
    replicate sets:
    [[replicates in time/concentration point 1],
     [replicates in time/concentration point 2],
     ...]
    :return seqToFit: a list of sequences will be used in fitting
    """

    import numpy as np

    def read_initPool(initDirc):
        '''
        Read the sequences detected in the initial pool, and return a dictionary of sequences
        :param initDirc: directory to the count file of initial pool
        :return: candidList: a dictionary of all sequences detected in the initial pool
        '''
        candidList = {}
        with open(initDirc) as file:
            next(file)
            next(file)
            next(file)
            for line in file:
                seq = line.strip().split()
                candidList[seq[0]] = {
                    'rawCounts': [int(seq[1])] + [np.nan for i in sampleDircList]
                }
        print('Candidate sequences imported from initial pool')
        return candidList

    def align_samples(candidList, sampleDircList, validSampleList):
        """
        Survey sequence counts in each k-seq sample and add raw counts of sequences in candidList to the list
        :param candidList: sequences detected in initial pool
        :param sampleDircList: directories to the k-seq sample count file
        :param validSampleList: A list of valid samples
        :return: an updated candidList with raw counts from valid k-seq samples
        """
        validSampleList = [rep for repSet in validSampleList for rep in repSet]
        for sampleIx, sampleDirc in enumerate(sampleDircList):
            if (sampleIx + 1) in validSampleList:
                print('Surveying sequences in sample %s ...' %sampleDirc)
                with open(sampleDirc) as file:
                    next(file)
                    next(file)
                    next(file)
                    for line in file:
                        seq = line.strip().split()
                        if seq[0] in candidList.keys():
                            candidList[seq[0]]['rawCounts'][sampleIx + 1] = int(seq[1])
        return candidList

    def filter_seq(candidList):
        """
        Filter out sequences that are not detected in any valid k-seq samples
        :param candidList
        :return validSeq: a list of valid sequences; number of seqs in original candidList, number of seqs in validSeq
        """
        def get_config(seq):
            return [sum([1 for rep in repSet if ~np.isnan(seq['rawCounts'][rep])]) for repSet in validSampleList]

        validSeq = []
        for seq in candidList.items():
            if np.nansum(seq[1]['rawCounts'][1:]) > 0:
                validSeq.append({
                    'seq': seq[0],
                    'rawCounts': seq[1]['rawCounts'],
                    'config': get_config(seq[1]),
                    'id': len(validSeq)
                })
        return (validSeq, len(candidList), len(validSeq))

    candidList = align_samples(candidList=read_initPool(initDirc),
                               sampleDircList=sampleDircList,
                               validSampleList=validSampleList)

    return filter_seq(candidList)

def get_normalized_fraction(seqToFit, sampleTotals, qFactors):
    """
    :param seqToFit: list of sequences will pass to fitting
    :param sampleTotals: list of total counts of reads in samples
    :param qFactors:
    :return:
    """
    import numpy as np

    for seq in seqToFit:
        frac = np.array(seq['rawCounts'])/sampleTotals*qFactors
        seq['reactedFrac'] = frac/frac[0]
    return seqToFit

def fitting(seq, xdata, maxFold=None, fitMtd='trf', ciEst=True, func=None, bsDepth=500):
    """
    :param seq: the sequence to fit
    :param xdata: the x-value, corresponding to the list of ALL samples
    :param maxFold: if the maximum reacted fraction can exceed the indicated value, default None
    :param fitMtd: fitting method to use
    :param ciEst: If the confidence interval will be estimated by bootstrapping
    :param func: func to fit, if None will fit to default exponential function
    :return: seq with fitting results
    """
    from scipy.optimize import curve_fit
    import numpy as np

    def exp_func(x, A, k):
        return A * (1 - np.exp(- 0.479 * 90 * k * x))  # BTO degradation adjustment and 90 minutes

    if func is None:
        func = exp_func
    ydata = np.array(seq['reactedFrac'][1:])
    if not(maxFold is None):
        ydata = np.array([min(yi, maxFold) for yi in ydata])
    valid = ~np.isnan(ydata)
    try:
        initGuess = [np.random.random(), np.random.random()]
        params, pcov = curve_fit(func, xdata=xdata[valid], ydata=ydata[valid],
                                 method=fitMtd, bounds=([0, 0], [1., np.inf]), p0=initGuess)
        x_ = xdata[valid]
        y_ = ydata[valid]
        yHat = func(x_, params[0], params[1])
        res = y_ - yHat
        pctRes = res / yHat
        ss_res = np.sum(res ** 2)
        ss_tot = np.sum((ydata[valid] - np.mean(ydata[valid])) ** 2)
        seq['r2'] = (1 - ss_res / ss_tot)
    except RuntimeError:
        params = [np.nan, np.nan]
    seq['params'] = [params[0], params[1], params[0]*params[1]]
    if ciEst:
        if (len(xdata[valid])>1)and(~np.isnan(params[0])):
            paramList = []
            for i in range(bsDepth):
                pctResResampled = np.random.choice(pctRes, replace=True, size=len(pctRes))
                yToFit = yHat + yHat * pctResResampled
                try:
                    initGuess = (np.random.random(), np.random.random())
                    params, pcov = curve_fit(func, x_, yToFit,
                                             method=fitMtd, bounds=([0, 0], [1., np.inf]), p0=initGuess)
                except:
                    params = [np.nan, np.nan]
                    print(x_, yToFit)
                paramList.append([params[0], params[1], params[0] * params[1]])
            seq['stdevs'] = np.nanstd(paramList, axis=0, ddof=1)
            seq['ci95'] = np.array([np.percentile(paramList, 2.5, axis=0), np.percentile(paramList, 97.5, axis=0)]).T
        else:
            seq['stdevs'] = [np.nan]
            seq['ci95'] = [np.nan]

    return seq



# fitting_main is not ready to use
def fitting_main(seqToFit=None, maxFold=None, fitMtd='trf', ciEst=True, func=None):
    import util
    import time
    import multiprocessing as mp

    timeInit = time.time()
    pool = mp.Pool(processes=8)
    if simuSet is None:
        simuSet = util.load_pickle('/mnt/storage/projects/ribozyme_predict/data/k_seq/fit_simu/simuSet_2_full.pkl')
    simuSet = pool.map(method_5_multi, simuSet)
    timeEnd = time.time()
    print('Process finished in %i s' % (timeEnd - timeInit))

    if not (simuSet is None):
        return simuSet
    else:
        pass
        # util.dump_pickle(simuSet,
        #              '/mnt/storage/projects/ribozyme_predict/data/k_seq/fit_simu/simuSet_1_method_4_parallel_res.pkl',
        #              log='10000 simulated data on [3,3,3,3] using method 5: bootstrap residues for 500 times, parallelized',
        #              overwrite=True)

    return seqToFit
