import numpy as np
import util
from scipy.optimize import curve_fit

def func(x, A, k):
    return A * (1-np.exp(-0.479 * 90 * k * x))

def mse(x, y, A, k): 
    y_ = func(x, A, k)
    return np.mean((y_-y)**2)

def re(k, A, err, rep, avg, smpl, size):
    return 'k%i_A%.1f_err%.1f_rep%i_avg%i_%s_s%i' %(k, A, err, rep, avg, smpl, size)

def random_data_fitting(k, A, err, xTrue, replicate=1, size=1000, average=True):
    # np.random.seed(23)
    
    yTrue = A * (1 - np.exp(-0.479 * 90 * k * xTrue))
    fittingRes = {
        'y_': [],
        'x_': None,
        'k': [],
        'kerr': [],
        'A': [],
        'Aerr': [],
        'kA': [],
        'kAerr': [],
        'r2': [],
        'mse': [],
        'mseTrue': []
    }
    
    for epochs in range(size):
        
        y = np.array([[np.max([np.random.normal(loc=yt, scale=yt*err), 10e-6]) for yt in yTrue] for _ in range(replicate)])
        x = np.array([xTrue for _ in range(replicate)])
        
        if average:
            y_ = np.mean(y, axis=0)
            x_ = np.mean(x, axis=0)
        else:
            y_ = np.reshape(y, y.shape[0]*y.shape[1])
            x_ = np.reshape(x, x.shape[0]*x.shape[1])
        try:
            popt, pcov = curve_fit(func, x_, y_, method='trf', bounds=([0, 0], [1., np.inf]))
        except RuntimeError:
            popt = [0,0]
        
        fittingRes['y_'].append(y_)
        if fittingRes['x_'] is None:
            fittingRes['x_'] = x_
        fittingRes['k'].append(popt[1])
        fittingRes['kerr'].append((popt[1]-k)/k)
        fittingRes['A'].append(popt[0])
        fittingRes['Aerr'].append((popt[0]-A)/A)
        fittingRes['kA'].append(popt[0]*popt[1])
        fittingRes['kAerr'].append((popt[0]*popt[1]-k*A)/(k*A))
    
        y = np.reshape(y, y.shape[0]*y.shape[1])
        x = np.reshape(x, x.shape[0]*x.shape[1])
        
        res = y - (1-np.exp(-0.479 * 90 * popt[1] * x)) * popt[0]
        ss_res = np.sum(res**2)
        ss_tot = np.sum((y-np.mean(y))**2)
        fittingRes['r2'].append(1-ss_res/ss_tot)
        
        fittingRes['mse'].append(mse(x_, y_, A=popt[0], k=popt[1]))
        fittingRes['mseTrue'].append(mse(x_, y_, A=A, k=k))
        
        if epochs%10 == 0:
            util.progress_bar(epochs/size)

    return fittingRes

if __name__=='__main__':
    
    fittingResults = {}
    A = 0.5
    kList = [1, 100, 10000]
    errList = [0.0, 0.2, 0.5, 1.0]
    sizeList = [i for i in range(50) if i>=1]
    rep = 1
    
    for k in kList:
        for err in errList:
            for size in sizeList:
                x = np.logspace(np.log10(0.000001), np.log10(0.00025), size)
                sample = re(A=A, k=k, err=err, avg=False, rep=1, size=size, smpl='log')
                print('Calculating %s' %sample)
                fittingResults[sample] = random_data_fitting(k, A, err, x, average=False, replicate=1)
                
    util.dump_pickle(fittingResults, '/mnt/storage/projects/ribozyme_predict/data/k_seq/fit_simu/smlp_num_2.pkl',
                    log='this is sumulated data of different number of samples varies on k, err, rep, repeated')

