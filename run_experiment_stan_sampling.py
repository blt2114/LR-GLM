import tensorflow as tf
tf.enable_eager_execution()
import ipdb

import utils, laplace, sampling, posterior, evaluation, experiments, stan_sampling
import pickle
import numpy as np

import argparse



base_dir ="./toy_1-17_stan/"
#base_dir ="./toy_check/"
#base_dir ="./toy_1-9_laplace/"

## Set parameters of dataset
max_variance, variance_decay_rate = 5., 1.05
only_evaluation = False
include_sampling = False; include_advi = False
include_fast_sampling = False
sketch = True

# Settings for experiments to Run
N, Ds = 2500, [500]#, 5000, 10000, 15000]
#n_reps=8; Ms = [5, 20, 50, 100, 250]
n_reps=1; Ms = [5, 20, 50, 100, 200, 400, 500]

# Parameter of low rank approximation (# of power iterations in randomized
# svd)
n_random_iters = 2
ftol=1e-8

parser = argparse.ArgumentParser(description='Run experiments with synthetic data',)
parser.add_argument('-Ds', metavar='Ds', type=int, nargs='+', default=Ds,
                    help='Dimensions of data to test')
parser.add_argument('-Ms', metavar='Ms', type=int, nargs='+', default=Ms,
                    help='Ranks of low rank approximations')
args = parser.parse_args()
Ds = args.Ds
print("Ds", Ds)


# If we are including sampling, compile stan code
if (include_sampling or include_advi) and not only_evaluation:
    full_sm = stan_sampling.stan_GLM()
else:
    full_sm = None

if include_fast_sampling and not only_evaluation:
    fast_sm = stan_sampling.stan_fastGLM()
else:
    fast_sm = None

# set prior over beta
for D in Ds:
    print("\n\nBeginning experiments for D=%d"%D)
    mu_beta, sigma_beta = np.zeros(D), np.ones(D)
    run="D=%05d-MaxVar=%0.02f-VarDecayRate=%0.02f-SVDnIters=%02d-N=%05d"%(D,
            max_variance, variance_decay_rate, n_random_iters, N)

    base_fn = base_dir+run
    log_fn = base_fn+".log"

    # Run experiments
    if not only_evaluation:
        experiments.run_experiments(
                N, D, Ms, max_variance, variance_decay_rate, mu_beta, sigma_beta,
                n_random_iters, base_fn, log_fn, n_reps, plot_2D=False,
                include_bias=False, include_sampling=include_sampling,
                include_advi=include_advi, include_fast_sampling=include_fast_sampling,
                sketch=sketch, ftol=ftol, full_sm=full_sm, fast_sm=fast_sm)

    ### Evaluation
    approximation_methods = ["Laplace", "Diagonal_Laplace", "Prior"]

    if include_sampling:
        approximation_methods = ["NUTS"] + approximation_methods

    if include_fast_sampling:
        for M in Ms:
            if M>D: continue
            approximation_methods.append("Fast_NUTS_(M=%d)"%M)

    if include_advi:
        #approximation_methods = ["ADVI_MF", "ADVI_FR"] + approximation_methods
        approximation_methods = ["ADVI_MF"] + approximation_methods

    for M in Ms:
        if M>D: continue
        approximation_methods.append("Fast_Laplace_(M=%d)"%M)

    if sketch:
        for M in Ms:
            if M>D: continue
            approximation_methods.append("Random_Laplace_(M=%d)"%M)

    ### Create style dictionary for plots
    style_dict = utils.style_dict(Ms, D)

    # Evaluate predictions (error and calibration)
    print("\tbeginning evaluation")
    for approx in approximation_methods:
        for rep in range(n_reps):

            # TODO: perhaps merge this file-io stuff into the call to 'evaluate_predictions'
            # Load in data
            data_fn = base_fn +"_data_rep=%d"%rep+ ".pkl"
            f = open(data_fn,'rb')
            data = pickle.load(f)
            f.close()

            fn = utils.method_to_fn(base_fn, approx, rep)
            f=open(fn, 'rb')
            approx_posterior = pickle.load(f)
            f.close()
            if "Fast_Lap" in approx or "Rand" in approx:
                approx_posterior.U = tf.cast(approx_posterior.U, tf.float64)
                approx_posterior.W = tf.cast(approx_posterior.W, tf.float64)

            fn = base_fn+"_"+approx+"_rep=%d_predictions.pkl"%rep
            #ipdb.set_trace()
            evaluation.evaluate_prediction(fn, approx_posterior, data, include_oos=True,
                include_calibration=True, include_bayes=True)

            fn = base_fn+"_"+approx+"_rep=%d_credset_calibration.pkl"%rep
            evaluation.evaluate_credible_set_calibration(fn, approx_posterior, data)

    for rep in range(n_reps):
        fn = base_fn+"_rep=%d_credset_calibration.png"%rep
        evaluation.plot_credible_set_calibration(fn, base_fn,
                approximation_methods, rep, style_dict)

    for (just_HMC, no_HMC) in [(True, False), (False, True)] if include_sampling else [(False, True)]:
        # pass copy of approximation methdos to avoid mutation
        evaluation.plot_error_and_NLL(base_fn, list(approximation_methods),
                style_dict, n_reps, bayes=True,
                just_HMC=just_HMC, no_HMC=no_HMC)
        evaluation.plot_calibration(base_fn, approximation_methods, style_dict,
                n_reps, bayes=True, just_HMC=just_HMC,
                no_HMC=no_HMC)

    # Evaluate posterior approximation quality
    # set baseline method and pop from list of approximations (to which we will compare it)
    if include_sampling:
        baseline_method = "NUTS"
    else:
        baseline_method = "Laplace"
    approx_methods_cpy = list(approximation_methods)
    approx_methods_cpy.pop(
        approx_methods_cpy.index(baseline_method)
    )
    style_dict['colors'][baseline_method] = 'k'

    evaluation.evaluate_posterior(base_fn, baseline_method,
            approx_methods_cpy, style_dict, n_reps=n_reps,
            plot_cov_spectral_error=True)
