def fitting_check(k, A, xTrue, y, size=100, average=True):
    np.random.seed(23)

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
        # initGuess= (np.random.random(), np.random.random()*k*100)
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