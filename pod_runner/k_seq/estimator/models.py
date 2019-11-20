
def byo_model(x, A, k):
    """
    Default kinetic model used in BYO k-seq estimator:
                    A * (1 - np.exp(- 0.479 * 90 * k * x))
    - 90: t, reaction time (min)
    - 0.479: alpha, degradation adjustment parameter for BYO in 90 min
    - k: kinetic coefficient
    - A: maximal conversion the self-aminoacylation ribozyme

    Args:
        x (`float`): predictor, concentration of BYO for each sample, needs have unit mol
        A (`float`)
        k (`float`)

    Returns:
        reacted fraction given the predictor x and parameter (A, k)
    """
    import numpy as np

    return A * (1 - np.exp(- 0.479 * 90 * k * x))  # BYO degradation adjustment and 90 minutes
