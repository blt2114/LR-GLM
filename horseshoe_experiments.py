
import pystan
import matplotlib.pyplot as plt
import numpy as np
import pickle
import timeit
import imp
import utils
import tensorflow as tf
tf.enable_eager_execution()

def save_model(model, fit, time, file_name):
    with open(file_name, "wb") as f:
        pickle.dump({'model': model, 'fit' : fit, 'time' : time}, f, protocol=-1)

def load_model(path):
    with open(path, "rb") as f:
        data_dict = pickle.load(f)
        fit = data_dict['fit']
        time = data_dict['time']
    return fit, time

def save_relevant(fit, time, file_name):
    with open(file_name, "wb") as f:
        pickle.dump({'beta': fit.extract()['beta'], 'time' : time, 'summary': fit.summary('beta')['summary']}, f, protocol=-1)

def load_relevant(path):
    with open(path, "rb") as f:
        data_dict = pickle.load(f)
        beta = data_dict['beta']
        time = data_dict['time']
        summary = data_dict['summary']
    return fit, time, summary

def make_stan_model(low_rank_approx=True):
    if low_rank_approx:
        fast_glm_fin_horse_code = """
        data {
          int<lower=1> N; // Number of data
          int<lower=1> M; // Projection dimension
          int<lower=1> D; // Number of covariates
          matrix[D, M] U; // Projection matrix
          matrix[N, M] barX; // Projected design matrix
          int<lower=0> y[N]; // labels
          real m0; // rough sparsity level
        }

        transformed data {
          // real m0 = 10;
          real slab_scale = 3;    // Scale for large slopes
          real slab_scale2 = square(slab_scale);
          real slab_df = 25;      // Effective degrees of freedom for large slopes
          real half_slab_df = 0.5 * slab_df;
        }

        parameters {
          vector[D] beta_tilde;
          vector<lower=0>[D] lambda;
          real<lower=0> c2_tilde;
          real<lower=0> tau_tilde;
          real<lower=0> sigma;
        }

        transformed parameters {
            vector[D] beta;
            vector[M] bar_beta;
            real tau0 = (m0 / (D - m0)) * (sigma / sqrt(1.0 * N));
            real tau = tau0 * tau_tilde; // tau ~ cauchy(0, tau0)

            // c2 ~ inv_gamma(half_slab_df, half_slab_df * slab_scale2)
            // Implies that marginally beta ~ student_t(slab_df, 0, slab_scale)
            real c2 = slab_scale2 * c2_tilde;

            vector[D] lambda_tilde =
              sqrt( c2 * square(lambda) ./ (c2 + square(tau) * square(lambda)) );

            // beta ~ normal(0, tau * lambda_tilde)
            beta = tau * lambda_tilde .* beta_tilde;
            bar_beta = U' * beta;
        }

        model {
          // tau ~ cauchy(0, sigma)
          // beta ~ normal(0, tau * lambda)

          beta_tilde ~ normal(0, 1);
          lambda ~ cauchy(0, 1);
          tau_tilde ~ cauchy(0, 1);
          c2_tilde ~ inv_gamma(half_slab_df, half_slab_df);
          sigma ~ normal(0, 2);

          y ~ bernoulli_logit(barX * bar_beta);
        }
        """
        return pystan.StanModel(model_code=fast_glm_fin_horse_code)
    else:
        fast_glm_fin_horse_code = """
        data {
          int<lower=1> N; // Number of data
          int<lower=1> D; // Number of covariates
          matrix[N, D] X; // Design matrix
          int<lower=0> y[N]; // labels
          real m0; // rough sparsity level
        }

        transformed data {
          // real m0 = 10;
          real slab_scale = 3;    // Scale for large slopes
          real slab_scale2 = square(slab_scale);
          real slab_df = 25;      // Effective degrees of freedom for large slopes
          real half_slab_df = 0.5 * slab_df;
        }

        parameters {
          vector[D] beta_tilde;
          vector<lower=0>[D] lambda;
          real<lower=0> c2_tilde;
          real<lower=0> tau_tilde;
          real<lower=0> sigma;
        }

        transformed parameters {
            vector[D] beta;
            real tau0 = (m0 / (D - m0)) * (sigma / sqrt(1.0 * N));
            real tau = tau0 * tau_tilde; // tau ~ cauchy(0, tau0)

            // c2 ~ inv_gamma(half_slab_df, half_slab_df * slab_scale2)
            // Implies that marginally beta ~ student_t(slab_df, 0, slab_scale)
            real c2 = slab_scale2 * c2_tilde;

            vector[D] lambda_tilde =
              sqrt( c2 * square(lambda) ./ (c2 + square(tau) * square(lambda)) );

            // beta ~ normal(0, tau * lambda_tilde)
            beta = tau * lambda_tilde .* beta_tilde;
        }

        model {
          // tau ~ cauchy(0, sigma)
          // beta ~ normal(0, tau * lambda)

          beta_tilde ~ normal(0, 1);
          lambda ~ cauchy(0, 1);
          tau_tilde ~ cauchy(0, 1);
          c2_tilde ~ inv_gamma(half_slab_df, half_slab_df);
          sigma ~ normal(0, 2);

          y ~ bernoulli_logit(X * beta);
        }
        """
        return pystan.StanModel(model_code=fast_glm_fin_horse_code)

if __name__ == "__main__":
    print("Compiling Stan Models")
    stan_model_low_rank = make_stan_model(low_rank_approx=True)
    stan_model_full_rank = make_stan_model(low_rank_approx=False)
    # N_arr = [250, 500, 1000]
    # D_arr = [100, 200, 500, 1000, 2000]
    N_arr = [100, 1000]
    D_arr = [50, 200]
    m0 = 10
    max_variance, variance_decay_rate = 5., 1.1
    n_random_iters = 2
    for N in N_arr:
        for D in D_arr:
            # Generate / Prepare data
            beta_true = np.zeros(D)
            beta_true[:m0] = 10
            mu_beta, sigma_beta = np.zeros(D, dtype=np.float32), np.ones(D, dtype=np.float32)
            print("Generating Data for N={0}, D={1}".format(N, D))
            data = utils.gen_all_data_and_splits(
                N, D, variance_decay_rate, mu_beta, sigma_beta, rotate=True,
                max_variance=max_variance, include_bias=True, beta=beta_true)
            _, X_train, Y_train = data[0]
            X_train = X_train.numpy()
            Y_train = np.array((1+Y_train.numpy())/2, np.int)
            np.save('../generated_data/X_N_{0}_D_{1}'.format(N, D), X_train)
            np.save('../generated_data/y_N_{0}_D_{1}'.format(N, D), Y_train)
            start = timeit.default_timer()
            print("Full-rank Run: N={0}, D={1}".format(N, D))
            exact_glm_dat = {
                    'N': N,
                    'D': D,
                    'X': X_train,
                    'y': Y_train,
                    'm0': m0
                            }
            full_rank_fit = stan_model_full_rank.sampling(data=exact_glm_dat, iter=2 * D, chains=4, verbose=True)
            end = timeit.default_timer()
            sampling_time = end - start
            save_relevant(full_rank_fit, sampling_time, '../saved_models/full_rank_N_{}_D_{}.pkl'.format(N, D))
            M_grid = [] #np.arange(20, np.min([D, N]), D // 5)
            for M in M_grid:
                print("Low-rank Run: N={0}, D={1}, M={2}".format(N, D, M))
                start = timeit.default_timer()
                XU_train, U = utils.low_rank_approximation(tf.constant(X_train), M, n_iter=n_random_iters)
                approx_glm_dat = {
                        'N': N,
                        'D': D,
                        'M': M,
                        'U': U,
                        'barX': XU_train,
                        'y': Y_train,
                        'm0': m0
                }
                low_rank_fit = stan_model_low_rank.sampling(data=approx_glm_dat, iter=2 * D, chains=4, verbose=True)
                end = timeit.default_timer()
                sampling_time = end - start
                save_relevant(low_rank_fit, sampling_time, '../saved_models/low_rank_N_{}_D_{}_M_{}.pkl'.format(N, D, M))
