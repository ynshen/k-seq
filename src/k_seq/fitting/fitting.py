def get_fitting_data(seq, sequence_set=None, black_list=None, na_data_exclude=True):
    """
    get the x and y corresponding to seq for fitting, from pd.DataFrame .reacted_frac_table
    :param seq: the r
    :return:
    """
    if type(seq) is str:
        seq = sequence_set.reacted_frac_table['seq']
    if black_list:
        black_list += [name if sequence_set.sample_info[name]['sample_type'] == 'input' for name in list(seq.index)]






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

    def func_default(x, A, k):
        return A * (1 - np.exp(- 0.479 * 90 * k * x))  # BTO degradation adjustment and 90 minutes

    if func is None:
        func = func_default
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