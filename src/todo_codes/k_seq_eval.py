"""
This module contain methods for k-seq data evaluation
"""
__author__ = "Yuning Shen"
__email__ = "yuningshen@ucsb.edu"





def get_seq_relaxed_count(sampleDirc, centerSeq, maxDist=10):
    '''
    1) Survey sequence counts in sample file within the distance of 0, 1, ..., maxDist
    2) Return a list of accumulated counts
    '''
    import Levenshtein

    with open(sampleDirc) as file:
        next(file)
        next(file)
        next(file)
        distanceList = []
        for line in file:
            seq = line.strip().split()
            distanceList.append([seq[0], Levenshtein.distance(centerSeq, seq[0]), int(seq[1])])
    relaxedCounts = [sum([seq[2] for seq in distanceList if seq[1] <= cutoff]) for cutoff in range(maxDist+1)]
    return relaxedCounts


def fitting_convergence_test(x, y, rep=10, func=None, retMod='PMAD'):
    import numpy as np
    from scipy.optimize import curve_fit

    def func_default(x, A, k):
        return A * (1 - np.exp(-0.479 * 90 * k * x))

    if retMod not in ['MAD', 'PMAD', 'Full']:
        print('ERROR: Please indicate valid return mod in (MAD, PMAD, Full).')
        return None

    np.random.seed(23)
    if not func:
        func = func_default
    fittingRes = {
        'y': y,
        'x': x,
        'k': [],
        'A': [],
        'kA': [],
        'r2': [],
    }

    for _ in range(rep):
        initGuess = (np.random.random(), np.random.random())
        try:
            popt, pcov = curve_fit(func, x, y,
                                   method='trf', bounds=([0, 0], [1., np.inf]),
                                   p0=initGuess)
        except RuntimeError:
            popt = [np.nan, np.nan]
        fittingRes['k'].append(popt[1])
        fittingRes['A'].append(popt[0])
        fittingRes['kA'].append(popt[0] * popt[1])

        res = y - func(x, popt[0], popt[1])
        ss_res = np.sum(res ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        fittingRes['r2'].append(1 - ss_res / ss_tot)

    if retMod == 'MAD':
        return {
            'k': np.median(np.absolute(fittingRes['k'] - np.median(fittingRes['k']))),
            'A': np.median(np.absolute(fittingRes['A'] - np.median(fittingRes['A']))),
            'kA': np.median(np.absolute(fittingRes['kA'] - np.median(fittingRes['kA']))),
            'r2': np.median(np.absolute(fittingRes['r2'] - np.median(fittingRes['r2'])))
        }
    elif retMod == 'PMAD':
        return {
            'k': np.median(np.absolute(fittingRes['k'] - np.median(fittingRes['k'])))/np.median(fittingRes['k']),
            'A': np.median(np.absolute(fittingRes['A'] - np.median(fittingRes['A'])))/np.median(fittingRes['A']),
            'kA': np.median(np.absolute(fittingRes['kA'] - np.median(fittingRes['kA'])))/np.median(fittingRes['kA']),
            'r2': np.median(np.absolute(fittingRes['r2'] - np.median(fittingRes['r2'])))/np.median(fittingRes['r2'])
        }
    elif retMod == 'Full':
        return fittingRes
