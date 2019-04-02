def random_data_generator(k, A, err, xTrue, replicate=1, average=False):
    '''
    Generate a set of data with given k, A and noise level
    '''

    # np.random.seed(23)

    yTrue = A * (1 - np.exp(-0.479 * 90 * k * xTrue))
    y_ = np.array(
        [[np.max([np.random.normal(loc=yt, scale=yt * err), 10e-6]) for yt in yTrue] for _ in range(replicate)])
    x_ = np.array([xTrue for _ in range(replicate)])

    return (x_, y_)