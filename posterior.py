import utils
import numpy as np
from scipy import stats
import tensorflow as tf
import pickle

class Posterior:
    """A superclass for posterior approximations (or exact posterior)

    This could be using a parametric representation of the exact posterior,
    samples approximating the exact posterior, or a single point estimate.
    """

    def __init__(self):
        pass

    def predict(self, X, bayes=False, logistic=True):
        """predict Makes a prediction for the data provided

        Args:
            X: design [N, D]
            bayes: True if to return an approximation to the posterior
                predictive.  Else just use the posterior mean
            logistic: if prediction is for logistic regression or linear
                regression.

        Returns parameters of a predictive distribution (log Bernoulli
            probabilities for logistic regression or mean and variance for
            linear regression)
        """
        pass

    def mean(self):
        """mean returns the posterior mean"""
        pass

    def mean_and_cov(self):
        """mean_and_cov returns the posterior mean and covariance"""
        pass

    def marginal_stdevs(self):
        """marginal_stdevs returns the posterior marginal standard deviations"""
        pass

    def credible_percentiles(self, beta):
        """credible_percentiles returns the percentiles of each dimension of
        beta within the marginal credible sets"""
        pass

    def credible_set_calibration(self, percentiles, n_bins=20):
        cum_counts = np.cumsum(np.histogram(percentiles, bins=20,
            range=(0,1))[0])
        cal_curve = cum_counts/cum_counts[-1]
        return cal_curve

    def pickle(self, fn):
        f=open(fn, 'wb')
        pickle.dump(self, f)
        f.close()

class MAP(Posterior):
    """MAP is a maximum a posterior estimate of parameters

    This is a subclass of Posterior, we consider it to be a point estimate of
    the posterior.
    """

    def __init__(self, beta_map, name, runtime):
        self.beta_map = beta_map
        self.name=name
        self.time=runtime

    def mean(self):
        return self.beta_map.numpy()

    def mean_and_cov(self):
        return self.beta_map.numpy(), np.zeros([self.beta_map.shape[0]])

    def marginal_stdevs(self):
        return np.zeros([self.beta_map.shape[0]])

    def predict(self, X, bayes=False, logistic=True):
        """predict returns the predictive distribution for the Xs provided

        if logistic regression this is vector of class log probabilities

        if linear regression, this is the mean and uncertainty in the
        function (the uncertainty is zero for MAP)

        """
        assert not bayes
        beta = self.beta_map

        # Linear regression
        if not logistic:
            # return mean
            return (X @ beta[:,None])[:,0], tf.zeros([0.]*X.shape[0])

        # Logistic regression
        return -tf.math.softplus(-tf.einsum('ND,D->N',X, beta))

class MCMC_approx(Posterior):
    """MCMC_approx is a posterior approximation of consisting of samples

    """
    def __init__(self, beta_samples, name, runtime):
        """__init__ initializes the MCMC_approx object

        samples is an [K,D] array of K, D-dimensional samples.
        """
        self.beta_samples = beta_samples
        self.name=name
        self.time=runtime

    def mean(self):
        beta_mean = tf.reduce_mean(self.beta_samples, axis=0)
        return beta_mean.numpy()

    def mean_and_cov(self):
        beta_mean = tf.reduce_mean(self.beta_samples, axis=0)
        beta_cov = tf.transpose(self.beta_samples-beta_mean[None])@(
                self.beta_samples-beta_mean[None])/float(int(self.beta_samples.shape[0]))
        return beta_mean.numpy(), beta_cov.numpy()

    def marginal_stdevs(self):
        if hasattr(self, "marginal_sds"): return self.marginal_sds
        self.marginal_sds = np.std(self.beta_samples.numpy(), axis=0)
        return self.marginal_sds

    def credible_percentiles(self, beta):
        K = int(self.beta_samples.shape[0])
        percentiles = [
                np.sum(self.beta_samples[:, d]<beta[d])/K
                for d in range(len(beta))
                ]
        return np.array(percentiles)

    def predict(self, X, bayes=False, logistic=True):
        betas = self.beta_samples
        if not bayes:
            betas = tf.reduce_mean(betas, axis=0, keep_dims=True)

        ## Linear regression
        if not logistic:
            # return mean and uncertainty
            return tf.reduce_mean(X @ tf.transpose(betas), axis=1)

        ## Logistic Regression
        # evaluate activations for each paramter
        a=tf.einsum('ND,SD->NS', X, betas)
        logistic_log_probs = utils.phi(tf.ones(a.shape, dtype=tf.float64), a)

        return tf.reduce_logsumexp(logistic_log_probs, axis=1) - tf.log(np.float64(int(betas.shape[0])))

class Laplace_approx(Posterior):
    """Laplace_approx is full rank Laplace approximation to the posterior

    """
    def __init__(self, mu_N, Sigma_N, name, runtime):
        self.name=name
        self.mu_N = mu_N
        self.Sigma_N = Sigma_N
        self.time=runtime

    def mean(self):
        return self.mu_N.numpy()

    def mean_and_cov(self):
        return self.mu_N.numpy(), self.Sigma_N.numpy()

    def marginal_stdevs(self):
        if hasattr(self, "marginal_sds"): return self.marginal_sds
        self.marginal_sds = np.sqrt(np.diag(self.Sigma_N.numpy()))
        return self.marginal_sds

    def credible_percentiles(self, beta):
        mean, sds = self.mean(), self.marginal_stdevs()
        return stats.norm.cdf((beta-mean)/sds)

    def predict(self, X, bayes=False, logistic=True):
        if not bayes:
            # Linear regression
            if not logistic:
                # return mean and uncertainty
                return tf.einsum('ND,D->N', X, self.mu_N)

            # Logistic Regression
            return -tf.math.softplus(-tf.einsum('ND,D->N', X, self.mu_N))

        # Use Bayesian approximation
        # Linear regression
        if not logistic:
            # return mean and uncertainty
            return tf.einsum('ND,D->N', X, self.mu_N), tf.reduce_sum((X @ self.Sigma_N)*X, axis=1)

        # Logistic Regression
        return utils.bayes_logistic_prob(X, self.mu_N, self.Sigma_N)

class Factorized_Gaussian_approx(Posterior):
    """Factorized_Gaussian_approx is a factorized Gaussian approximation to
    the posterior.  This is used for the Diagonal Laplace approximation and for
    the prior.

    """
    def __init__(self, mu_N, Sigma_N, name, runtime):
        """here Sigma_N is diagonal """
        self.mu_N = mu_N
        self.Sigma_N = Sigma_N
        self.name=name
        self.time=runtime
        assert len(Sigma_N.shape)==1

    def mean(self):
        return self.mu_N.numpy()

    def mean_and_cov(self):
        return self.mu_N.numpy(), np.diag(self.Sigma_N.numpy())

    def marginal_stdevs(self):
        return np.sqrt(self.Sigma_N.numpy())

    def credible_percentiles(self, beta):
        mean, sds = self.mean(), self.marginal_stdevs()
        return stats.norm.cdf((beta-mean)/sds)

    def predict(self, X, bayes=False, logistic=True):
        if not bayes:
            # Linear regression
            if not logistic:
                # return mean
                return tf.einsum('ND,D->N', X, self.mu_N)

            # Logistic Regression
            return -tf.math.softplus(-tf.einsum('ND,D->N',X, self.mu_N))

        # Use Bayesian approximation
        # Linear regression
        if not logistic:
            # return mean and uncertainty
            return tf.einsum('ND,D->N', X, self.mu_N), tf.reduce_sum(X**2 *
                    self.Sigma_N[None], axis=1)

        # Logistic Regression
        return utils.bayes_logistic_prob(X, self.mu_N, self.Sigma_N)

class Fast_Laplace_approx(Posterior):
    """Fast_Laplace_approx is low rank Laplace approximation to the posterior

    """
    def __init__(self, mu_N, Sigma_beta, U, W, name, runtime):
        """
        To avoid an manipulation of the DxD covariance matrix we use the
        more concise representation:
        Sigma_n = Sigma_beta - (Sigma_beta U) W (U^T Sigma_beta),
        where W = (U^T Sigma_beta U - H^{-1})^-1
        """
        self.mu_N = mu_N
        self.Sigma_beta = Sigma_beta
        self.U = U
        self.W = W
        self.name=name
        self.time=runtime

    def mean(self):
        return self.mu_N.numpy()

    def mean_and_cov(self):
        """mean_and_cov returns the approximated posterior mean and
        covariance.

        Explictly compute full covariance as
        Sigma_n = Sigma_beta - (Sigma_beta U) W (U^T Sigma_beta),

        Returns mean and covariance as numpy arrays
        """
        Sigma_beta = self.Sigma_beta
        W, U = self.W, self.U
        Sigma_N = tf.diag(Sigma_beta) - (Sigma_beta[:,None]*U)@W@(tf.transpose(U)*Sigma_beta[None])
        return self.mu_N.numpy(), Sigma_N.numpy()

    def marginal_stdevs(self):
        if hasattr(self, "marginal_sds"): return self.marginal_sds
        Sigma_beta = self.Sigma_beta
        W, U = self.W.numpy(), self.U.numpy()
        marginal_var = lambda d: Sigma_beta[d] -Sigma_beta[d]*U[d].dot(W).dot(U[d])*Sigma_beta[d]
        self.marginal_sds = np.array([np.sqrt(marginal_var(d)) for d in range(len(Sigma_beta))])
        return self.marginal_sds

    def credible_percentiles(self, beta):
        mean, sds = self.mean(), self.marginal_stdevs()
        return stats.norm.cdf((beta-mean)/sds)

    def predict(self, X, bayes=False, logistic=True):
        """predict forms predition using an approximation to the approximate
        posterior predictive distribution

        """
        if not bayes:
            # Linear regression
            if not logistic:
                # return mean and uncertainty
                return tf.einsum('ND,D->N', X, self.mu_N)

            # Logistic Regression
            return -tf.math.softplus(-tf.einsum('ND,D->N', X, self.mu_N))

        Sigma_beta = self.Sigma_beta
        W, U = self.W, self.U

        # Use Bayesian approximation
        # Linear regression
        if not logistic:
            # return mean and uncertainty
            return (X @self.mu_N[:,None])[:, 0], \
                    tf.reduce_sum(X*Sigma_beta[None]*X, axis=1) - \
                    tf.reduce_sum( (U*Sigma_beta[None] @ W ) *
                            tf.transpose(U)*Sigma[:, None], axis=1)

        # Logistic Regression
        return utils.bayes_logistic_prob_fast(X, self.mu_N, U, Sigma_beta, W)

    def pickle(self, fn):
        f=open(fn, 'wb')
        self.W = tf.cast(self.W, dtype=tf.float32)
        self.U = tf.cast(self.U, dtype=tf.float32)
        pickle.dump(self, f)
        f.close()

