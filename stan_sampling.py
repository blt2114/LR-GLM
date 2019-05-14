import numpy as np
import pandas as pd
from time import time
import pystan
import tensorflow as tf

def stan_GLM():
    # adapted from https://betanalpha.github.io/assets/case_studies/bayes_sparse_regression.html
    glm_code = """
    data {
      int<lower=1> N; // Number of data
      int<lower=1> D; // Number of covariates
      matrix[N, D] X; // Design matrix
      int<lower=0> y[N]; // labels
      real<lower=0> sigma;
    }

    parameters {
      vector[D] beta;
    }

    model {
      beta ~ normal(0, sigma);
      y ~ bernoulli_logit(X * beta);
    }
    """
    start = time()
    sm_glm = pystan.StanModel(model_code=glm_code)
    end = time()
    print("compilation time: ", end-start, "seconds")
    return sm_glm


def stan_fastGLM():
    # FAST GLM
    # adapted from https://betanalpha.github.io/assets/case_studies/bayes_sparse_regression.html
    fast_glm_code = """
    data {
      int<lower=1> N; // Number of data
      int<lower=1> D; // Number of covariates
      int<lower=1> M; // Projected dimension
      matrix[D, M] U; // Projection matrix
      matrix[N, M] barX; // Projected design matrix
      int<lower=0> y[N]; // labels
      real<lower=0> sigma;
    }

    parameters {
      vector[D] beta;
    }


    transformed parameters {
      vector[M] bar_beta = U' * beta;
    }

    model {
      beta ~ normal(0, sigma);
      y ~ bernoulli_logit(barX * bar_beta);
    }
    """
    start = time()
    sm_fastGLM = pystan.StanModel(model_code=fast_glm_code)
    end = time()
    print("compilation time: ", end-start, "seconds")
    return sm_fastGLM


def sample_from_model(sm, dat, n_iter=1000):
    """sample_from_model runs the NUTS using stan

    Args:
        sm: the stan model
        dat: the data dictionary with the stan data inputs
        n_iter: number of iterations to run in each chain, half are tossed
            as warm-up samples.


    Returns:
        samples drawn, time for sampling, mean effective sample size and
            mean Rhat
    """
    st = time()
    fit = sm.sampling(data=dat, iter=n_iter, chains=4)
    end = time()
    sampling_time = end-st
    s = fit.summary()
    summary = pd.DataFrame(s['summary'], columns=s['summary_colnames'], index=s['summary_rownames'])
    mean_ESS = np.mean(summary["n_eff"])
    mean_Rhat = np.mean(summary["Rhat"])
    samples = fit.extract()['beta']
    return samples, sampling_time, mean_ESS, mean_Rhat

def advi_with_model(sm, dat, algorithm='meanfield'):
    """advi_with_model runs the advi using stan

    Args:
        sm: the stan model
        dat: the data dictionary with the stan data inputs
        algorithm: meanfield or full rank

    Returns:
        samples drawn and time for inferece
    """
    st = time()
    fit = sm.vb(data=dat, algorithm=algorithm)
    end = time()
    advi_time = end-st
    samples = np.array(fit['sampler_params'][:-1]).T
    return samples, advi_time
