import sys
if '/home/yuning/Work/k-seq/src/' not in sys.path:
    sys.path.append('/home/yuning/Work/k-seq/src/')

import util
import time
import k_seq.data.simu
import k_seq.

def func(x, A, k):
    return A * (1 - np.exp(-0.479 * 90 * k * x))


def fitting_check(k, A, xTrue, y, size=100, average=True):
    # np.random.seed(23)

    fittingRes = {
        'y_': None,
        'x_': None,
        'k': [],
        'kerr': [],
        'A': [],
        'Aerr': [],
        'kA': [],
        'kAerr': [],
        'mse': [],
        'mseTrue': [],
        'r2': []
    }

    if average:
        y_ = np.mean(y, axis=0)
        x_ = np.mean(xTrue, axis=0)
    else:
        y_ = np.reshape(y, y.shape[0] * y.shape[1])
        x_ = np.reshape(xTrue, xTrue.shape[0] * xTrue.shape[1])

    for epochs in range(size):
        initGuess = (np.random.random(), np.random.random())

        try:
            popt, pcov = curve_fit(func, x_, y_, method='trf', bounds=([0, 0], [1., np.inf]), p0=initGuess)
        except RuntimeError:
            popt = [np.nan, np.nan]

        if fittingRes['y_'] is None:
            fittingRes['y_'] = y_
        if fittingRes['x_'] is None:
            fittingRes['x_'] = x_
        fittingRes['k'].append(popt[1])
        fittingRes['kerr'].append((popt[1] - k) / k)
        fittingRes['A'].append(popt[0])
        fittingRes['Aerr'].append((popt[0] - A) / A)
        fittingRes['kA'].append(popt[0] * popt[1])
        fittingRes['kAerr'].append((popt[0] * popt[1] - k * A) / (k * A))

        fittingRes['mse'].append(mse(x_, y_, A=popt[0], k=popt[1]))
        fittingRes['mseTrue'].append(mse(x_, y_, A=A, k=k))

        res = y_ - (1 - np.exp(-0.479 * 90 * popt[1] * x_)) * popt[0]
        ss_res = np.sum(res ** 2)
        ss_tot = np.sum((y_ - np.mean(y_)) ** 2)
        fittingRes['r2'].append(1 - ss_res / ss_tot)

    return fittingRes


def calculate_convergence(k, A, err):
    pctRange = []
    for i in range(10):
        x_, y_ = random_data_generator(k=k, A=A, err=err, xTrue=xTrue, average=True, replicate=3)
        fittingRes = fitting_check(k=k, A=A, xTrue=x_, y=y_, size=10)
        pctRange.append(fittingRes)
    return pctRange


if __name__ == '__main__':
    timeStart = time.time()

    xSeries = [2e-6, 1e-5, 2e-5, 2.5e-4, 1e-3, 5e-3]
    x = np.array([2e-6, 2e-5, 2.5e-4, 5e-3])
    print('Selected concetration for [BTO]: %r' %xTrue)
    kValues = np.logspace(-1, 4, 100)
    AValues = np.linspace(0, 1, 101)[1:]
    err = 0.0

    convMtx = []
    for ix, k in enumerate(kValues):
        convMtx.append([calculate_convergence(k, A, err) for A in AValues])
        util.progress_bar(ix/100)

    util.dump_pickle(convMtx,
                     '/mnt/storage/projects/ribozyme_predict/data/k_seq/fit_simu/convergence_scan_err%.2f_x%.1e-%.1e_allRes.pkl' %(err, xTrue[0], xTrue[-1]),
                     log='Scan the convergence of k (res 100), A (res 100), each (k,A) sampled 10 datasets for 10 fittings with selected concentration %s, all fitting results preserved' %xTrue,
                     overwrite=True)
    print('Code finish running in %f' %(time.time()-timeStart))