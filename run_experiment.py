import tensorflow as tf
tf.enable_eager_execution()

import utils, laplace, sampling, posterior, evaluation, experiments
import pickle
import numpy as np

base_dir ="./toy_1-8/"

## Set parameters of dataset
N, Ds = 2500, [10, 50, 100, 250, 500]#, 5000, 10000, 15000]
max_variance, variance_decay_rate = 5., 1.05
only_evaluation = False
include_sampling = True
sketch = True

# Parameter of low rank approximation (# of power iterations in randomized
# svd)
n_random_iters = 2
ftol=1e-6

# Settings for experiments to Run
n_reps=3; Ms = [5, 50, 100, 250, 500, 1000, 1500]

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
                sketch=sketch, ftol=ftol)

    ### Evaluation
    approximation_methods = ["Laplace", "Diagonal_Laplace", "Prior"]

    if include_sampling:
        approximation_methods = ["NUTS"] + approximation_methods
        for M in Ms:
            if M>=D: continue
            approximation_methods.append("Fast_HMC_(M=%d)"%M)

    for M in Ms:
        if M>=D: continue
        approximation_methods.append("Fast_Laplace_(M=%d)"%M)

    if sketch:
        for M in Ms:
            if M>=D: continue
            approximation_methods.append("Sketch_Laplace_(M=%d)"%M)

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

            fn = base_fn+"_"+approx+"_rep=%d_predictions.pkl"%rep
            evaluation.evaluate_prediction(
                fn, approx_posterior, data, include_oos=True,
                include_calibration=True, include_bayes=True)

            fn = base_fn+"_"+approx+"_rep=%d_credset_calibration.pkl"%rep
            evaluation.evaluate_credible_set_calibration(fn, approx_posterior, data)

    for rep in range(n_reps):
        fn = base_fn+"_rep=%d_credset_calibration.png"%rep
        evaluation.plot_credible_set_calibration(fn, base_fn,
                approximation_methods, rep, style_dict)


    for include_bayes  in [True]:
        for (just_HMC, no_HMC) in [(True, False), (False, True)] if include_sampling else [(False, True)]:
            # pass copy of approximation methdos to avoid mutation
            evaluation.plot_error_and_NLL(base_fn, list(approximation_methods),
                    style_dict, n_reps, bayes=include_bayes,
                    just_HMC=just_HMC, no_HMC=no_HMC)
            evaluation.plot_calibration(base_fn, approximation_methods, style_dict,
                    n_reps, bayes=include_bayes, just_HMC=just_HMC,
                    no_HMC=no_HMC)

    # Evaluate posterior approximation quality
    # set baseline method and pop from list of approximations (to which we will compare it)
    if include_sampling:
        baseline_method = "HMC"
    else:
        baseline_method = "Laplace"
    approx_methods_cpy = list(approximation_methods)
    approx_methods_cpy.pop(
        approx_methods_cpy.index(baseline_method)
    )
    style_dict['colors'][baseline_method] = 'k'

    #ipdb.set_trace()
    evaluation.evaluate_posterior(base_fn, baseline_method,
            approx_methods_cpy, style_dict, n_reps=n_reps)
