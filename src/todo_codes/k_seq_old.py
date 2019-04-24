def count_profile(countTable, rndToCount):
    for seq in countTable:
        seq.append('True')
        seq.append(0)
        for rndBatch in rndToCount:
            counter = 0
            for rnd in rndBatch:
                if seq[1][rnd] != 0:
                    counter = 1
                else:
                    seq[-2] = False
            seq[-1] += counter
    return countTable

def readSeqs(dirc, countCutoff=1, normalize=True, quantFactor=1):
    # read all sequences from dirc and normalized counts on total counts and quantFactor
    # input fileType: a list of sequences followed by count numbers, with three lines of header info

    allSeqCounts = {}

    print('Calculating %s ...' % dirc)

    with open(dirc) as f:
        # get uniques and totals from input file heads
        line0 = next(f)
        uniqueSplit = [elem for elem in line0.strip().split()]
        # print(uniqueSplit)
        uniques = float(uniqueSplit[-1])
        line1 = next(f)
        totalSplit = [elem for elem in line1.strip().split()]
        # print(totalSplit)
        totals = float(totalSplit[-1])
        next(f)

        # calculate [normalized] counts
        i = 0
        for lineRead in f:
            line = [elem for elem in lineRead.strip().split()]
            i += 1
            if int(line[1]) >= countCutoff:  # count the seq above cutOff
                if normalize:
                    allSeqCounts[line[0]] = float(line[1]) / totals / quantFactor
                else:
                    allSeqCounts[line[0]] = int(line[1])

    return (allSeqCounts, uniques, totals)


def alignCounts(masterCounts, roundCounts, rnd, totalRndNum, initialize=False):
    if initialize:
        for seq in roundCounts:
            counts = [0] * totalRndNum
            counts[0] += roundCounts[seq]
            masterCounts.append([seq, counts])
    else:
        for seq in masterCounts:
            if seq[0] in roundCounts:
                seq[1][rnd] += roundCounts[seq[0]]  # exist in round align; not exist in round 0
            else:
                seq[1][rnd] = 0

    return masterCounts


def countAll(primDirc, otherDirc, primMin=1, otherMin=1, normList=[]):
    # ---------------------- count masterDirc -----------------------
    masterCounts = []

    round0Counts, round0uniques, round0totals = readSeqs(primDirc, countCutoff=primMin,
                                                         normalize=True, quantFactor=normList[0])
    masterUniques = [round0uniques]
    masterTotals = [round0totals]

    # ------------- count otherLoc ---------------------

    totalRndNum = len(otherDirc) + 1  # number of rounds + 1 (firstLoc)

    masterCounts = alignCounts(masterCounts, round0Counts, 0, totalRndNum, initialize=True)
    masterSet = {}
    for seq in masterCounts:
        masterSet[seq[0]] = 0  # initial masterSets

    thisRnd = 0
    for loc in otherDirc:
        thisRnd += 1
        (seqs, uniqs, tots) = readSeqs(loc, countCutoff=otherMin, normalize=True, quantFactor=normList[thisRnd])
        masterUniques.append(uniqs)
        masterTotals.append(tots)
        masterCounts = alignCounts(masterCounts, seqs, thisRnd, totalRndNum)

    return masterCounts, masterUniques, masterTotals


def data_fitting(seqToFit, rndsToAvg, eqnType, maxOne=False, maxFold=None, fitMtd='trf', devEstimate=False):
    import util
    from scipy.optimize import curve_fit
    import numpy as np

    def func_0(x, A, k):
        return A * (1 - np.exp(- k * x)) # no adjustment

    def func_1(x, A, k):
        return A * (1 - np.exp(- 0.479 * k * x)) # BTO degradation adjustment

    def func_2(x, A, k):
        return A * (1 - np.exp(- 0.479 * 90 * k * x))  # BTO degradation adjustment and 90 minutes

    def get_rnd_avg(seq):
        avgs = []
        for rndBatch in rndsToAvg:
            rndBatchCounts = []
            for rnd in rndBatch:
                if seq[1][rnd] > 0:
                    if maxFold:
                        rndBatchCounts.append(min(maximum_fold, (seq[1][rnd] / float(seq[1][0]))))
                    else:
                        rndBatchCounts.append(seq[1][rnd] / float(seq[1][0]))  # percentage of reacted
            if maxOne:
                avgs.append(min(1, np.average(rndBatchCounts)))
            else:
                avgs.append(np.average(rndBatchCounts))
        return avgs

    def get_rnd_samples(seq):
        np.random.seed(23)
        samples = []
        flag = False
        for i in range(3):
            sample = []
            for rndBatch in rndsToAvg:
                concen = [seq[1][rnd] for rnd in rndBatch if seq[1][rnd] > 0]
                if len(concen) >= 2:
                    flag = True
                if len(concen) > 0:
                    sample.append(np.random.choice(concen, 1)[0] / float(seq[1][0]))
                else:
                    sample.append(np.nan)
            samples.append(sample)
        return flag, samples

    xdata = np.array([0.00025, 0.00005, 0.00001, 0.000002])
    seqNum = len(seqToFit)
    checkpoint = progress_checkpoint(seqNum)

    for ix, seq in enumerate(seqToFit):
        ydata = np.array(get_rnd_avg(seq))
        valid = ~np.isnan(ydata)
        try:
            if eqnType == 0:
                params, pcov = curve_fit(func_0, xdata=xdata[valid], ydata=ydata[valid],
                                         method=fitMtd, bounds=([0, 0], [1., np.inf]))
            elif eqnType == 1:
                params, pcov = curve_fit(func_1, xdata=xdata[valid], ydata=ydata[valid],
                                         method=fitMtd, bounds=([0, 0], [1., np.inf]))
            elif eqnType == 2:
                params, pcov = curve_fit(func_2, xdata=xdata[valid], ydata=ydata[valid],
                                         method=fitMtd, bounds=([0, 0], [1., np.inf]))
        except RuntimeError:
            params = [0, 0]

        seq.append([params[0], params[1], params[0]*params[1]])
        seq.append(ydata)

        if eqnType == 0:
            res = ydata[valid] - (1 - np.exp(- params[1] * xdata[valid])) * params[0]
        elif eqnType == 1:
            res = ydata[valid] - (1 - np.exp(- 0.479 * params[1] * xdata[valid])) * params[0]
        elif eqnType == 2:
            res = ydata[valid] - (1 - np.exp(- 0.479 * 90 * params[1] * xdata[valid])) * params[0]
        ss_res = np.sum(res ** 2)
        ss_tot = np.sum((ydata[valid] - np.mean(ydata[valid])) ** 2)
        seq.append(1 - ss_res/ss_tot)

        if devEstimate:
            yflag, ydataSet = get_rnd_samples(seq)
            if yflag:
                paramsSet = []
                for ydata in ydataSet:
                    ydata = np.array(ydata)
                    valid = ~np.isnan(ydata)
                    try:
                        if eqnType == 0:
                            params, pcov = curve_fit(func_0, xdata=xdata[valid], ydata=ydata[valid],
                                                     method=fitMtd, bounds=([0, 0], [1., np.inf]))
                        elif eqnType == 1:
                            params, pcov = curve_fit(func_1, xdata=xdata[valid], ydata=ydata[valid],
                                                     method=fitMtd, bounds=([0, 0], [1., np.inf]))
                        elif eqnType == 2:
                            params, pcov = curve_fit(func_2, xdata=xdata[valid], ydata=ydata[valid],
                                                     method=fitMtd, bounds=([0, 0], [1., np.inf]))
                    except RuntimeError:
                        params = [0, 0]
                    paramsSet.append([params[0], params[1], params[0]*params[1]])
                seq.append(True)
                seq.append(np.std(np.array(paramsSet), axis=0))
                seq.append(paramsSet)
                seq.append(ydataSet)
            else:
                seq.append(False)
                seq.append([np.nan, np.nan, np.nan])
                seq.append(np.nan)
                seq.append(np.nan)


        if ix in checkpoint:
            progress_bar(ix / seqNum)

    return seqToFit
