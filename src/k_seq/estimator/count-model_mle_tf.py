#!/home/yuning/.pyenv/versions/k-seq/bin python

from yuning_util.dev_mode import DevMode
import argparse
dev_mode = DevMode(pkg='k-seq')
dev_mode.on()
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)


# shape indication: s: number of sample, s_i: number of input sample, s_r: number of reacted sample,
# m: number of sequences
def reparameterization(theta, psi, phi):
    """Reparameterize p0, k, and a to avoid constraints
    m is the number of seqs

    Args:
        theta: shape [m] tensor
        psi: shape [m] tensor
        phi: shape [m] tensor

    Returns:
        p0: shape [m] tensor
        k: shape [m] tensor
        A: shape [m] tensor
    """

    p0 = tf.math.softmax(theta)
    k = tf.math.exp(psi)
    a = tf.exp(phi) / (1 + tf.exp(phi))

    return p0, k, a


def kinetic_model(p0, k, a, c, alpha, t):
    """A pseudo-first order kinetic model
    input intial pool and parameters of observation
    output pool composition for all observations shape [s_r, m],
        s_r is number of reacted samples, m is number of sequences

    Args:
        p0: shape [m] tensor
        k: shape [m] tensor
        a: shape [m] tensor
        c: shape [s_r] tensor
        alpha, t: scalar parameter

    """

    # expand c(shape [s_r]) to c_repeat([s_r, m])
    c_rep = tf.tile(tf.expand_dims(c, -1), [1, tf.shape(p0)[0]])

    frac = tf.multiply(1 - tf.exp(-alpha * t * tf.multiply(c_rep, k)), a)
    p = tf.multiply(frac, p0)

    return tf.divide(p, tf.expand_dims(tf.reduce_sum(input_tensor=p, axis=1), -1))


def loss(params_1d, c, alpha, t, y, mask):
    """Negative Log likelihood as loss, reduced on all observations

    Args:
        params_1d: shape [3m] parameters contains [theta (p0), psi (k), and phi (a)]
        c: shape [s] constant contains substrates concentrations
        alpha, t: scalar parameters
        y: shape [s, m] tensor contains counts information

    """

    params = tf.reshape(params_1d, [3, -1])
    theta = params[0]
    psi = params[1]
    phi = params[2]

    # convert parameter
    p0, k, a = reparameterization(theta, psi, phi)

    # calculate reacted pool p
    probs_reacted = kinetic_model(p0, k, a, tf.boolean_mask(c, mask), alpha, t)
    # calculate input pool p0
    probs_input = tf.tile(tf.expand_dims(p0, 0), [np.sum(~mask), 1])

    # total counts
    n = tf.reduce_sum(y, axis=-1)

    # get Multinomial distributions
    total_count = tf.boolean_mask(n, mask)
    model_reacted = tfd.Multinomial(total_count=tf.boolean_mask(n, mask), probs=probs_reacted)
    total_count = tf.boolean_mask(n, ~mask)
    model_input = tfd.Multinomial(total_count=tf.boolean_mask(n, ~mask), probs=probs_input)

    # return averaged likelihood
    return -tf.reduce_sum(model_reacted.log_prob(tf.boolean_mask(y, mask))) - \
           tf.reduce_sum(model_input.log_prob(tf.boolean_mask(y, ~mask)))


def loss_and_gradient(params_1d, alpha, t, c, y, mask):
    """pass a value and gradient object to tfp.optimizer.lbfgs_minimize,
    conditioned on the given data: (c, y)

    Args:
        params_1d: [3m] shape parameters
        y: []
    """

    value, gradient = tfp.math.value_and_gradient(
        lambda params: loss(params, c=c, y=y, alpha=alpha, t=t, mask=mask),
        params_1d
    )

    return value, tf.reshape(gradient, [-1])


def get_model(ctrl_vars, counts, pool_mask, alpha=0.479, t=90, method='lbfgs', init_method='uniform',
          max_iterations=1000, num_correction_pairs=10, tolerance=1e-8, parallel_iterations=1):
    """a fitting node to conduct fitting

    Args:
        ctrl_vars: controlled variable, x, here is concentration (c), shape [s]
        counts: dependent variable, y, shape [s, m]
        pool_mask: boolean mask indicate input pool (False) or reacted pool (True), shape [s]
    """

    if init_method.lower() in ['unif', 'uniform']:
        # initialize params [theta, psi, phi] with zero
        param_init = tf.zeros([3 * tf.shape(counts)[1]])
    elif init_method.lower() in ['average', 'avg']:
        # initialize theta with log(counts + 0.01), psi, phi with zero
        p0_init = counts[~pool_mask].sum(axis=0) + 0.01
        theta_init = np.log(p0_init)
        param_init = tf.convert_to_tensor(
            np.concatenate(
                [theta_init, np.zeros_like(theta_init), np.zeros_like(theta_init)]
            ), dtype=np.float32
        )
        # param_init shape [3m]
    opt = tfp.optimizer.lbfgs_minimize(
        lambda params: loss_and_gradient(params, c=ctrl_vars, y=counts, alpha=alpha, t=t, mask=pool_mask),
        initial_position=param_init,
        max_iterations=max_iterations,
        num_correction_pairs=num_correction_pairs,
        tolerance=tolerance,
        x_tolerance=0,
        f_relative_tolerance=0,
        parallel_iterations=parallel_iterations,
        stopping_condition=None,
        name='lbfgs_minimier'
    )
    return opt


def fit(ctrl_var, counts):

    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as sess:
        pool_mask = ctrl_var > 0  # True for reacted sample, False for input sample
        mle_model = get_model(ctrl_vars=ctrl_var, counts=counts, pool_mask=pool_mask, init_method='average',
                              max_iterations=args.max_iteration, num_correction_pairs=args.num_correction_pairs,
                              tolerance=args.tolerance, parallel_iterations=args.parallel_iteration)
        results = sess.run(mle_model)
        print(f'Convergence: {results.converged}')
    return results


def main(output_path, simu_data_path=None, seq_table=None, table_name=None):
    from src.k_seq import CountData
    from src.k_seq import Timer, FileLogger
    from pathlib import Path

    if not Path(output_path).exists():
        Path(output_path).mkdir(parents=True)

    if simu_data_path is not None:
        count_data = CountData.from_simu_path(simu_data_path)
        ctrl_var = count_data.ctrl_vars.loc['c'].astype(np.float32).values
        counts = count_data.count.to_numpy().astype(np.float32).T
    elif seq_table is not None:
        from src.k_seq import read_pickle
        seq_table = read_pickle(seq_table)
        ctrl_var = seq_table.x_values.values
        counts = getattr(seq_table, table_name).to_numpy().astype(np.float32).T

    with Timer(), FileLogger(output_path + '/estimator.log'):
        results = fit(ctrl_var=ctrl_var, counts=counts)

    from src.k_seq import dump_pickle
    dump_pickle(obj=results, path=output_path + '/results.pkl')


def parse_param(results, index):
    import pandas as pd
    import numpy as np
    from scipy.special import softmax

    params = results.position
    params = np.reshape(params, [3, -1])
    theta = params[0]
    psi = params[1]
    phi = params[2]

    p0 = softmax(theta)
    k = np.exp(psi)
    a = np.exp(phi) / (1 + np.exp(phi))

    return pd.DataFrame({'p0': p0, 'k': k, 'a': a}, index=index)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run count model MLE using tensorflow')
    parser.add_argument('--output_path', '-o', type=str,
                        help='Path to save outputs')
    parser.add_argument('--seq_table', '-t', default=None,
                        help='path to pickled seqTable object')
    parser.add_argument('--table_name', '-n', default=None,
                        help='table name if use seq_table')
    parser.add_argument('--input_pools', '-i', default=None,
                        help="Name for input samples")
    parser.add_argument('--simu_data_path', '-s', default=None,
                        help='Path to the folder of simulated data')
    parser.add_argument('--solver', '-m', default='SCS',
                        help='Which solver to use, support SCS and MOSEK')
    parser.add_argument('--notes', default=None,
                        help='Optional notes for data')
    parser.add_argument('--max_iteration', default=1000, type=int,
                        help='Max iteration for solver')
    parser.add_argument('--num_correction_pairs', type=int, default=10,
                        help='[BFGS] Number of s, r pairs used in lBFGS')
    parser.add_argument('--tolerance', type=float, default=1e-8,
                        help='[BFGS] Tolerance of supremum norm gradient')
    parser.add_argument('--parallel_iteration', type=int, default=1,
                        help='[BFGS] Number of parallel iteration to compute in parallel')

    args = parser.parse_args()

    main(output_path=args.output_path,
         simu_data_path=args.simu_data_path,
         seq_table=args.seq_table,
         table_name=args.table_name)


