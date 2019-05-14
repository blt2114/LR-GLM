import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

def HMC(log_prob_fn, D, n_samples=1000, n_leapfrog_steps=8,
        step_size=np.float64(0.02),
        n_chains=4, n_burnin=50):
    """HMC runs hamiltonian monte carlo using a specificied log prob
    function.

    This is agnostic to the underlying density (e.g. sparsity
    inducing vs Gaussian prior or full model/low-rank approximaion)


    Args:
        log_prob_fn: target log probability density (function of beta)
        D:  dimension of parameter (input to log_prob_fn)

    Returns:
        samples, acceptance rate
    """
    # Construct TFP McMC Sampler
    current_state = tf.zeros([n_chains, D], dtype=tf.float64)
    [x_samples], kernel_results = tfp.mcmc.sample_chain(
        num_results=n_samples,
        num_burnin_steps=n_burnin,
        current_state=[current_state],
        kernel=tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=log_prob_fn,
                step_size=step_size,
                num_leapfrog_steps=n_leapfrog_steps))
    acceptance_rate = tf.reduce_mean(tf.to_float(kernel_results.is_accepted))
    # Compute effective sample as minimum effective sample size across
    # dimensions.
    # x_samples is of shape [N samples, n chains, D]
    assert x_samples.shape[1]==n_chains

    # combine samples across chains
    x_samples = tf.reshape(x_samples, [-1, D])

    ess = tfp.mcmc.effective_sample_size(x_samples)
    ess = np.min(ess)
    return x_samples, acceptance_rate, ess

def NUTS(log_prob_fn, D, n_samples=1000, step_size=np.float64(0.02), n_burnin=50):
    """NUTS runs the no uturn sampler using the specificied log prob
    function.

    Args:
        log_prob_fn: target log probability density (function of beta)
        D:  dimension of parameter (input to log_prob_fn)

    Returns:
        samples, acceptance rate
    """
    # Construct TFP McMC Sampler
    n_accept, samples = 0, []
    current_state = tf.constant(tf.zeros([D], dtype=tf.float64))
    log_prob_fn_new = lambda beta: log_prob_fn(beta[None])[0]

    print("beginning warm-up")
    for s in range(n_samples+n_burnin):
        if s==n_burnin: print("now collecting samples")
        [
            [next_state], _, _,
        ] = tfp.no_u_turn_sampler.kernel(
                    current_state=[current_state],
                    target_log_prob_fn=log_prob_fn_new,
                    step_size=[step_size])
        if s >= n_burnin:
            if current_state!=next_state:
                n_accept += 1
            samples.append(next_state.numpy())
        current_state = next_state
    acceptance_rate = n_accept/len(samples)
    # Compute effective sample as minimum effective sample size across
    # dimensions.

    x_samples = np.array(samples)
    ess = tfp.mcmc.effective_sample_size(x_samples)
    ess = np.min(ess)
    x_samples = tf.constant(x_samples)
    return x_samples, acceptance_rate, ess

def low_rank_NUTS(XU, U, Y, log_prior, phi):
    """low_rank_NUTS runs NUTS using the likelihood approximation.

    Args:
        XU: projected data (N by M)
        U: projection matrix with orthonormal columns, so U^TU=I (D by M)
        Y: Labels (N)
        log_prior: function takes parameters & returns log prior (R^D --> R)
        phi: GLM mapping function, maps (x_i^T\beta, y_i) --> log p(y_i|x_i, y)

    Returns:
        samples, acceptance rate, effective sample size
    """
    log_prob_fn = lambda beta: tf.reduce_sum(phi(
        Y[:, None], tf.einsum('NM,MC->NC', XU, tf.einsum('DM,CD->MC', U, beta))
            ),
                axis=0) + log_prior(beta)
    return NUTS(log_prob_fn, U.shape[0])

def low_rank_HMC(XU, U, Y, log_prior, phi, n_chains=1):
    """low_rank_HMC runs HMC using the likelihood approximation.

    Args:
        XU: projected data (N by M)
        U: projection matrix with orthonormal columns, so U^TU=I (D by M)
        Y: Labels (N)
        log_prior: function takes parameters & returns log prior (R^D --> R)
        phi: GLM mapping function, maps (x_i^T\beta, y_i) --> log p(y_i|x_i, y)

    Returns:
        samples, acceptance rate, effective sample size
    """
    log_prob_fn = lambda beta: tf.reduce_sum(phi(
        Y[:, None], tf.einsum('NM,MC->NC', XU, tf.einsum('DM,CD->MC', U, beta))
            ),
                axis=0) + log_prior(beta)
    return HMC(log_prob_fn, U.shape[0], n_chains=n_chains)

def full_HMC(X, Y, log_prior, phi, n_chains=1):
    """full_HMC runs HMC on GLM

    Args:
        X: Design (N by D)
        log_prior: function takes parameters & returns log prior (R^D --> R)
        phi: GLM mapping function, maps (x_i^T\beta, y_i) --> log p(y_i|x_i, y)

    Returns:
        samples, acceptance rate, effective sample size
    """
    log_prob_fn = lambda beta: tf.reduce_sum(
            phi(Y[:, None], tf.einsum('ND,CD->NC', X, beta)), axis=0) + log_prior(beta)
    return HMC(log_prob_fn, X.shape[1], n_chains=n_chains)

def full_NUTS(X, Y, log_prior, phi):
    """full_NUTS runs NUTS on GLM

    Args:
        X: Design (N by D)
        log_prior: function takes parameters & returns log prior (R^D --> R)
        phi: GLM mapping function, maps (x_i^T\beta, y_i) --> log p(y_i|x_i, y)

    Returns:
        samples, acceptance rate, effective sample size
    """
    log_prob_fn = lambda beta: tf.reduce_sum(
            phi(Y[:, None], tf.einsum('ND,CD->NC', X, beta)), axis=0) + log_prior(beta)
    return NUTS(log_prob_fn, X.shape[1])
