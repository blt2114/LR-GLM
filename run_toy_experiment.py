import tensorflow as tf
tf.enable_eager_execution()

import utils, laplace, sampling, posterior, evaluation, experiments
import pickle
import numpy as np

base_dir ="./toy_2D_1-23/"

## Set parameters of dataset
N, D = 50, 3
max_variance, variance_decay_rate = 1.0, 40.
include_sampling = False
include_bias = True

# Parameter of low rank approximation (# of power iterations in randomized
# svd)
n_random_iters = 50

# Settings for experiments to Run
Ms = [1]

# set prior over beta
print("\n\nBeginning experiments for D=%d"%D)
mu_beta, sigma_beta = np.zeros(D, dtype=np.float64), np.ones(D,
        dtype=np.float64)
run="D=%05d-MaxVar=%0.02f-VarDecayRate=%0.02f-SVDnIters=%02d-N=%05d"%(D,
        max_variance, variance_decay_rate, n_random_iters, N)

base_fn = base_dir+run
log_fn = base_fn+".log"

# Run experiments and plot
import ipdb; ipdb.set_trace()
experiments.run_experiments(
        N, D, Ms, max_variance, variance_decay_rate, mu_beta, sigma_beta,
        n_random_iters, base_fn, log_fn, n_reps=1, plot_2D=True,
        include_bias=include_bias, include_sampling=include_sampling,
        include_fast_sampling=include_sampling,
        regenerate_data=False)
