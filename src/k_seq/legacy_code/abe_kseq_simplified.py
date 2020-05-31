"""
Created on Apr 24, 2017

This script was reorganized by Yuning Shen

@author: abepres
"""

import numpy as np
from scipy.optimize import curve_fit
import time
import util

def readSeqs(loc, fileType='counts', cutOff=3, norm2=1, whiteList={}):
    #fileType: 'counts' refers to a list of sequences followed by count numbers, with three lines of header information

    allSeqCounts = {}
    initTime = time.time()
    print('Import sequences from %s...' %loc, end=" ")
    with open(loc) as f:
        if fileType == 'counts':
            # get unique and total number of reads
            line0 = next(f)
            uniqueSplit = [elem for elem in line0.strip().split()]
            uniques = float(uniqueSplit[-1])
            line1 = next(f)
            totalSplit = [elem for elem in line1.strip().split()]
            totals = float(totalSplit[-1])
            next(f)
            i = 0
            for lineRead in f:
                line = [elem for elem in lineRead.strip().split()]
                i += 1
                if int(line[1]) >= cutOff:
                    if whiteList:
                        if line[0] in whiteList:
                            allSeqCounts[line[0]] = float(line[1])/totals/norm2
                    else:
                        allSeqCounts[line[0]] = float(line[1])/totals/norm2

    print('finished in %i sec' %(time.time()-initTime))
    return (allSeqCounts, uniques, totals)
    #a list of all sequences we want to search over, and their appearance

def alignCounts(masterCounts, roundCounts, rnd, maxRnds, initialize=False):

    if initialize:
        for seq in roundCounts:
            counts = [0]*maxRnds   # Notice: initialize with 0
            counts[0] += roundCounts[seq]
            masterCounts.append([seq, counts])
    else:
        for seq in masterCounts:
            if seq[0] in roundCounts.keys():
                seq[1][rnd] += roundCounts[seq[0]]

    return masterCounts

def readAll(firstLoc, otherLoc, firstMin, otherMin, normList=[]):

    masterCounts = []
    maxRnds = len(otherLoc) + 1
    (round0Counts, round0uniques, round0totals) = readSeqs(firstLoc, cutOff=firstMin, norm2 = normList[0],
                                                           whiteList=[])
    masterUniques = [round0uniques]
    masterTotals = [round0totals]
    masterCounts = alignCounts(masterCounts, round0Counts, 0, maxRnds, initialize=True)
    masterSet = set([seq[0] for seq in masterCounts])
    print("%i sequences found in input pool" %len(round0Counts))
    thisRnd = 0
    for loc in otherLoc:
        thisRnd += 1
        if normList:
            (seqs, uniqs, tots) = readSeqs(loc, cutOff=otherMin, norm2=normList[thisRnd], whiteList=masterSet)
        masterUniques.append(uniqs)
        masterTotals.append(tots)
        masterCounts = alignCounts(masterCounts, seqs, thisRnd, maxRnds)

    return (masterCounts, masterUniques, masterTotals)

def filter_seqs(tempCounts):
    print("%i candidate sequences detected" %(len(tempCounts)))
    validRnds = []
    for concen in testAvg[1]:
        validRnds += concen
    mask = [True if i in validRnds else False for i in range(25)]
    validSeqs = [seq for seq in tempCounts if not(np.sum(np.array(seq[1])[mask])==0)]
    print("%i valid sequences found" %(len(validSeqs)))
    util.dump_pickle(data=validSeqs,
                      dirc='/mnt/storage/projects/ribozyme_predict/k_seq/abe_validSeqs.pkl',
                      log='The %i valid sequences (being detected at least once in valid k-seq samples) generated from Abe\'s code, (%i seqs filtered by previous abe_validSeqs.pkl for unvalid samples). A list of sequences with [seq, [normalized counts in each samples, starting from input]]' %(len(validSeqs), len(tempCounts)),
                      overwrite=True)
    return validSeqs

def printCounts(outLoc, counts, uniqs, tots, firstLoc, separator=',', fitAvg=[], compact=False):
    # main function for estimator
    # ----------define type of calculation and output heading

    if compact == 2:
        line0 = 'X' + separator + 'Abun ' + firstLoc
    if fitAvg:
        line0 += separator + 'L by avg' + separator + 'k by avg' + separator + 'L stdev' + separator + 'k stdev'
    line0 += '\n'
    line1 = 'Unique'
    if compact == 2:
        line1 += separator + str(uniqs[0])
    else:
        for uniq in uniqs:
            line1 += separator + str(uniq)
    line1 += '\n'
    line2 = 'Total'
    if compact == 2:
        line2 += separator + str(tots[0])
    else:
        for tot in tots:
            line2 += separator + str(tot)
    line2 += '\n'
    if fitAvg:
        xdata = np.array(fitAvg[0])
        rndsToAvg = fitAvg[1]
        rndsToError = fitAvg[2]
        if len(fitAvg[0]) != len(fitAvg[1]):
            print("length mismatch error")
    fittingRes = []
    with open(outLoc,'w') as outFile:
        outFile.write(line0)
        outFile.write(line1)
        outFile.write(line2)

        def func(x, L, k):
            return L * (1 - np.exp(-k * x))

        counter_o_counts = 0
        print("Fitting %i sequences..." %len(counts))

        for seq in counts:
            counter_o_counts += 1
            if compact == 2:
                lineOut = seq[0] # seq's seq
                lineOut += separator + str(seq[1][0]) # Seq's abundance in input
            fittingRes.append(
                {
                    'seq': seq[0],
                    'normedAmount':seq[1]
                }
            )
            if fitAvg:
                avgs = []
                for timePoint in rndsToAvg:
                    timePtCounts = []
                    for rnd in timePoint:
                        if seq[1][0]>0:
                            timePtCounts.append(seq[1][rnd]/(float(seq[1][0])))
                        else:
                            print("Error: count in input smaller than 1")
                            timePtCounts.append(1)
                    avgs.append(np.average(timePtCounts))
                ydata = np.array(avgs)
                fittingRes[-1]['yValue'] = ydata
                try:
                    popt, pcov = curve_fit(func, xdata, ydata, method='trf', bounds=(0, [1., np.inf]))
                except RuntimeError:
                    popt = [0,0]
                fittingRes[-1]['params'] = [popt[0], popt[1], popt[0]*popt[1]]

                lineOut += separator + str(popt[0]) + separator + str(popt[1])

                Lset = []
                kset = []

                if popt[1] > 0.0001:
                    #arbitrary threshold for k
                    fittingRes[-1]['stddevYValue'] = []
                    for rndSet in rndsToError:
                        setVals = []
                        for rnd in rndSet:
                            if seq[1][0]>0:
                                setVals.append(seq[1][rnd]/float(seq[1][0]))
                            else:
                                setVals.append(1)
                        popt, pcov = curve_fit(func, xdata, np.array(setVals), method='trf', bounds=(0, [1., np.inf]))
                        Lset.append(popt[0])
                        kset.append(popt[1])
                        fittingRes[-1]['stddevYValue'].append(np.array(setVals))
                    fittingRes[-1]['stddevSeries'] = [np.array(Lset), np.array(kset), np.array(Lset)*np.array(kset)]
                    fittingRes[-1]['stddev'] = [np.std(param) for param in fittingRes[-1]['stddevSeries']]
                    lineOut += separator + str(np.std(Lset)) + separator + str(np.std(kset))
                else:
                    lineOut += separator + '0' + separator + '0'
                    fittingRes['stddev'] = [np.nan]
            lineOut += '\n'
            outFile.write(lineOut)
    util.dump_pickle(data=fittingRes,
                      dirc='/mnt/storage/projects/ribozyme_predict/k_seq/abe_fittingRes.pkl',
                      log='Fitting results from Abe\'s code',
                      overwrite=True)


testList = ['counts-1A.txt','counts-1B.txt','counts-1C.txt','counts-1D.txt','counts-1E.txt','counts-1F.txt',
            'counts-2A.txt','counts-2B.txt','counts-2C.txt','counts-2D.txt','counts-2E.txt','counts-2F.txt',
            'counts-3A.txt','counts-3B.txt','counts-3C.txt','counts-3D.txt','counts-3E.txt','counts-3F.txt',
            'counts-4A.txt','counts-4B.txt','counts-4C.txt','counts-4D.txt','counts-4E.txt','counts-4F.txt']
testNorm = [0.0005,
            0.023823133, 0.023823133, 0.023823133, 0.023823133, 0.023823133, 0.023823133,
            0.062784812, 0.062784812, 0.062784812, 0.062784812, 0.062784812, 0.062784812,
            0.159915207, 0.159915207, 0.159915207, 0.159915207, 0.159915207, 0.159915207,
            0.53032596, 0.53032596, 0.53032596, 0.53032596, 0.53032596, 0.53032596]
testAvg = ([0.00025, 0.00005, 0.00001, 0.000002],
           [[1, 2, 3, 4, 5, 6],  # A list of valid rounds
            [7, 8, 10, 11, 12],
            [13, 14, 15],
            [19, 20, 21, 23]],
           [[1, 7, 13, 19],    # Three manually selected replicates to estimate the standard deviation
            [2, 8, 14, 20],
            [3, 12, 15, 21]])
print('Calculation starting...')
root = '/mnt/storage/projects/ribozyme_predict/count-file/'
(tempCounts, un, to) = readAll(root + 'R5c-counts.txt', [root + dirc for dirc in testList], 1, 1, normList=testNorm)
validSeq = filter_seqs(tempCounts)
# validSeq = util.load_pickle(dirc='/mnt/storage/projects/ribozyme_predict/k_seq/abe_validSeqs.pkl')
printCounts(outLoc='allseqs-name-fit.csv', counts=validSeq, uniqs=un, tots=to, firstLoc='R5c-counts.txt', fitAvg=testAvg, compact=2)
