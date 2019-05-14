import posterior, laplace, sampling, utils, evaluation, stan_sampling
import pickle
import numpy as np
import os
from time import time
import tensorflow as tf
tf.enable_eager_execution()

def run_experiments(N, D, Ms, max_variance, variance_decay_rate, mu_beta,
        sigma_beta, n_random_iters, base_fn, log_fn, n_reps=3,
        plot_2D=False, include_bias=False, include_sampling=True, include_fast_sampling=True,
        include_advi=False, regenerate_data=False, beta=None, sketch=False,
        rotate=True, data=None, ftol=1e-6, full_sm=None, fast_sm=None):
    """run_experiments a number of synthetic data experiments, generating
    data, fitting models and saving approximations.


    Args:
        N: number of datapoints for training set
        D: dimension of model
        Ms: ranks of low rank approximations (list)
        max_variance: maximum variance of any dimension in the design matrix
           distribution.
        variance_decay_rate: rate of decay of the variance in each dimension
            of the design matrix.
        mu_beta: prior mean
        sigma_beta: prior variance
        n_random_iters: number of passes through the data in the randomized
            svd
        base_fn: base filename for files to save with approximations
        log_fn: file to log to
        n_reps: number of replicates
        include_bias: set true to include bias (by having first column of X
            be all 1's)
        plot_2D: set true if D==3 (2 + bias) to generate plots of data and
            decision boundaries.
        regenerate_data: Set true to always generate data
        beta: parameter (sampled if None)

    No returns
    """
    # perform several replicates
    for rep in range(n_reps):
        utils.log("\n\nBeginning Rep %d\n"%rep, log_fn)
        data_fn = base_fn +"_data_rep=%d"%rep+ ".pkl"
        if data is None and os.path.isfile(data_fn) and not regenerate_data:
            f= open(data_fn,'rb')
            data = pickle.load(f)
            f.close()
        elif data is None or rep != 0:
            utils.log("\tGenerating data", log_fn)
            data = utils.gen_all_data_and_splits(
                N, D, variance_decay_rate, mu_beta, sigma_beta,
                rotate=rotate, max_variance=max_variance,
                include_bias=include_bias, beta=beta)
            # dump generated data to pickle
            f= open(data_fn,'wb')
            pickle.dump(file=f,obj=data)
            f.close()
        _, X_train, Y_train = data[0]

        ## First add prior
        method_name = "Prior"
        eps = 0.1 # set to small positive value so time can be plotted on log scale
        prior = posterior.Factorized_Gaussian_approx(
            tf.constant(mu_beta), tf.constant(sigma_beta), method_name, runtime=eps)
        prior.pickle(utils.method_to_fn(base_fn, method_name,rep))

        ## find MAP solution
        utils.log("\tfinding map", log_fn)
        method_name = "MAP"
        start = time()
        beta_map = laplace.MAP(X_train, Y_train, sigma_beta, utils.phi,
                utils.dphi_da, verbose=False, ftol=ftol)
        end = time()
        time_map = end-start
        approx = posterior.MAP(beta_map, method_name, time_map)
        approx.pickle(utils.method_to_fn(base_fn, method_name,rep))
        utils.log("\tTime: %0.02f seconds"%time_map, log_fn)

        # Diagonal Laplace
        utils.log("diagonal laplace", log_fn)
        method_name = "Diagonal_Laplace"
        start = time()
        diag_mu_N, diag_Sigma_N = laplace.diag_laplace(
            X_train, Y_train, sigma_beta, utils.phi, utils.dphi_da,
            utils.d2phi_da2, beta_map=beta_map, ftol=ftol)
        end = time()
        time_diag = time_map + (end-start)
        approx = posterior.Factorized_Gaussian_approx(
            diag_mu_N, diag_Sigma_N, method_name, time_diag)
        approx.pickle(utils.method_to_fn(base_fn, method_name,rep))
        utils.log("\tTime: %0.02f seconds"%time_diag, log_fn)

        # Full Laplace
        utils.log("full laplace", log_fn)
        method_name = "Laplace"
        start = time()
        full_mu_N, full_Sigma_N = laplace.laplace(
            X_train, Y_train, sigma_beta, utils.phi, utils.dphi_da,
            utils.d2phi_da2, beta_map=beta_map, ftol=ftol)
        end = time()
        time_laplace = time_map + (end-start)
        approx = posterior.Laplace_approx(
            full_mu_N, full_Sigma_N, method_name, time_laplace)
        approx.pickle(utils.method_to_fn(base_fn, method_name,rep))
        utils.log("\tTime: %0.02f seconds"%time_laplace, log_fn)


        ## Sampling (HMC)
        # define prior (accomodate multiple chains)
        if include_sampling:
            log_prior = lambda beta: -0.5*tf.reduce_sum(tf.log(2.*np.pi*sigma_beta)) \
                - 0.5*tf.reduce_sum(beta**2/sigma_beta[None], axis=1)

            # run NUTS
            method_name="NUTS"
            utils.log("full NUTS (STAN)", log_fn)
            dat = glm_toy_dat = {
                'N':N, 'D':D,
                'X': X_train.numpy(), 'y': np.array((1+Y_train.numpy())/2, np.int),
                'sigma': sigma_beta[0]
            }
            samples, time_NUTS, mean_ESS, mean_Rhat = stan_sampling.sample_from_model(full_sm, dat)
            samples = tf.constant(samples)
            approx = posterior.MCMC_approx(samples, method_name, time_NUTS)
            approx.pickle(utils.method_to_fn(base_fn, method_name,rep))
            utils.log("\tTime: %0.02f seconds"%time_NUTS, log_fn)
            utils.log("\tRhat: %f\n\teESS: %f"%(mean_Rhat, mean_ESS), log_fn)

        # ADVI (both MF and full rank)
        if include_advi:
            log_prior = lambda beta: -0.5*tf.reduce_sum(tf.log(2.*np.pi*sigma_beta)) \
                - 0.5*tf.reduce_sum(beta**2/sigma_beta[None], axis=1)
            dat = glm_toy_dat = {
                'N':N, 'D':D,
                'X': X_train.numpy(), 'y': np.array((1+Y_train.numpy())/2, np.int),
                'sigma': sigma_beta[0]
            }

            # run ADVI mean field
            method_name="ADVI_MF"
            utils.log("ADVI_MF", log_fn)
            samples, time_ADVI_MF = stan_sampling.advi_with_model(full_sm,
                    dat, algorithm="meanfield")
            samples = tf.constant(samples)
            approx = posterior.MCMC_approx(samples, method_name, time_ADVI_MF)
            approx.pickle(utils.method_to_fn(base_fn, method_name,rep))
            utils.log("\tTime: %0.02f seconds"%time_ADVI_MF, log_fn)

            ## run ADVI full rank
            #method_name="ADVI_FR"
            #utils.log("ADVI_FR", log_fn)
            #samples, time_ADVI_FR = stan_sampling.advi_with_model(full_sm,
            #        dat, algorithm="fullrank")
            #samples = tf.constant(samples)
            #approx = posterior.MCMC_approx(samples, method_name, time_ADVI_FR)
            #approx.pickle(utils.method_to_fn(base_fn, method_name,rep))
            #utils.log("\tTime: %0.02f seconds"%time_ADVI_FR, log_fn)

        # Low rank approximations
        utils.log("Beginning low rank approximations", log_fn)
        for M in Ms:
            if M > D: continue
            # Truncated SVD
            utils.log("\tM=%d\t Performing Truncated SVD"%M, log_fn)
            start = time()
            XU_train, U = utils.low_rank_approximation(X_train, M, n_iter=n_random_iters)
            end = time()
            svd_time = end - start
            utils.log("\t\tTime: %0.02f seconds"%svd_time, log_fn)

            # Fast Laplace
            utils.log("\tfast laplace", log_fn)
            method_name = "Fast_Laplace_(M=%d)"%M
            start = time()
            fast_mu_N, fast_W = laplace.fast_laplace(
                XU_train, U, Y_train, sigma_beta, utils.phi, utils.dphi_da,
                utils.d2phi_da2, verbose=True, ftol=ftol)
            end = time()
            time_low_rank_laplace = end-start
            fast_Sigma_N = tf.diag(sigma_beta) - tf.diag(sigma_beta)@U@fast_W@tf.transpose(U)@tf.diag(sigma_beta)
            approx = posterior.Fast_Laplace_approx(
                fast_mu_N, sigma_beta, U, fast_W, method_name, time_low_rank_laplace + svd_time)
            approx.pickle(utils.method_to_fn(base_fn, method_name,rep))
            utils.log("\t\tTime: %0.02f seconds"%time_low_rank_laplace, log_fn)

            # Fast HMC
            if include_fast_sampling:
                method_name="Fast_NUTS_(M=%d)"%M
                utils.log("\tfast NUTS (STAN)", log_fn)
                dat = {
                    'N': N, 'D': D, 'M': M,
                    'U': U, 'barX': XU_train.numpy(),
                    'y': np.array((1+Y_train.numpy())/2, np.int),
                    'sigma': sigma_beta[0],
                }
                start = time()
                samples, time_NUTS, mean_ESS, mean_Rhat = stan_sampling.sample_from_model(fast_sm, dat)
                samples = tf.constant(samples)
                end = time()
                time_low_rank_NUTS = svd_time + (end-start)
                approx = posterior.MCMC_approx(samples, method_name, time_low_rank_NUTS)
                approx.pickle(utils.method_to_fn(base_fn, method_name,rep))
                utils.log("\tTime: %0.02f seconds"%time_NUTS, log_fn)
                utils.log("\tRhat: %f\n\teESS: %f"%(mean_Rhat, mean_ESS), log_fn)

            if sketch:
                utils.log("\tsketching Laplace", log_fn)
                start = time()
                S = np.random.normal(size=[D, M], scale=1./np.sqrt(D))
                S = tf.constant(S)
                XS_train = X_train@S
                method_name = "Random_Laplace_(M=%d)"%M
                sketch_mu_N, sketch_W = laplace.fast_laplace(
                    XS_train, S, Y_train, sigma_beta, utils.phi, utils.dphi_da, utils.d2phi_da2)
                end = time()
                time_sketch_laplace = end-start
                sketch_Sigma_N = tf.diag(sigma_beta) - tf.diag(sigma_beta)@S@sketch_W@tf.transpose(S)@tf.diag(sigma_beta)
                approx = posterior.Fast_Laplace_approx(
                    sketch_mu_N, sigma_beta, S, sketch_W, method_name, time_sketch_laplace)
                approx.pickle(utils.method_to_fn(base_fn, method_name,rep))
                utils.log("\t\tTime: %0.02f seconds"%time_sketch_laplace, log_fn)

        if plot_2D:
            evaluation.plot2D_posterior_samples(X_train, Y_train, beta_map,
                    mu_beta, sigma_beta, diag_mu_N, diag_Sigma_N, full_mu_N,
                    full_Sigma_N, fast_mu_N, fast_Sigma_N,
                    full_beta_samples=None, fast_beta_samples=None,
                    base_fn=base_fn, has_bias=include_bias)
