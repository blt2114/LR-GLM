"""Logistic Regression utilities

This module contains several basic utilities for performing and evaluating
inference methods for Bayesian logistic regression.  These include
generating synthetic data and evaluating the logistic function as well as
its first and second derivatives.

"""
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
import seaborn as sns
from scipy import stats
import tensorflow as tf

# Currently in `contributed' / Experimental but soon to be default
tfe = tf.contrib.eager

def method_to_fn(base_fn, method_name, rep):
    return base_fn+"_"+method_name+"_rep=%d.pkl"%rep

def log(msg, fn, also_stdout=True):
    f = open(fn, 'a')
    f.write(msg+"\n")
    f.close()
    if also_stdout: print(msg)

def low_rank_approximation(X, M, n_iter=2, verbose=False):
    """low_rank_approximation performs a randomized SVD to create a low rank
    approximation of the data.

    A truncated svd of X is performed:
        X^T \\approx U Lambda V^T

    The data is projected as:
        XU (N by M)


    Args:
        X: design matrix (X by D)
        M: rank of approximation
        n_iter: number of power iterations (Halko 2009)

    Returns:
        the projected data, and projection (XU and U)
    """
    if X.shape[0] == M or X.shape[1] == M:
        if verbose: print("performing full SVD")
        _, _, UT = np.linalg.svd(X, full_matrices=False) # U is shape [N, M=D]
        U = tf.constant(UT.T)
    else:
        skl_svd = TruncatedSVD(n_components=M, n_iter=n_iter)
        if verbose: print("performing truncated SVD")
        skl_svd.fit(X)
        U = tf.transpose(tf.constant(skl_svd.components_))
        if verbose: print("performing X \\dot V.T")
    XU = tf.einsum('ND,DM->NM',X, U)
    return XU, U

def bayes_logistic_prob_fast(X, mu_n, U, Sigma_beta, W):
    """ Posterior predictive logistic regression probability.  Uses probit approximation
        to the logistic regression sigmoid. Also has overflow prevention via exponent truncation.

    adapted from: https://github.com/Valassis-Digital-Media/bayes_logistic

    The posterior covariance is approximated as:
    Sigma_N = Sigma_beta - (Sigma_beta * U) * W * (U^T * Sigma_beta)

    This provides that for each observation x_i, the uncertainty in the
    activation is:
    sig2_a = x_i^T Sigma_beta x_i - (x_i^T Sigma_beta U) W (U^T Sigma_beta x_i)

    For a collection of N observations, X an [N, D] array, and a diagonal
    prior variance (required here) we efficiently compute this all sig2_a's
    as:
    Sig2_a = sum_d X_{:,d}^2 *Sigma_beta_{d,d} -
        [(X*Sigma_beta [None]) dot U]  dot W [ U.T dot (Sigma_beta[:,
        None]*X.T)]

    We compute this effieciently as:
    Sig2_a = reduce_sum(X**2 * Sigma_beta[None], axis=1) -
           reduce_sum(A@W * A, axis=1)
    where  A = X*Sigma_beta[None] @ U


    Returns the log predicted probability of y=1
    """
    assert len(Sigma_beta.shape) ==1

    # look at eigenvalues of W
    eigs= np.linalg.eigvals(W.numpy())
    assert np.sum(W.numpy()!=W.numpy().T)==0

    # unmoderated argument of exponent
    mu_a = tf.einsum('ND,D->N', X, mu_n)

    Sig2_a = tf.reduce_sum(X**2 * Sigma_beta[None], axis=1)
    A = (X*Sigma_beta[None]) @ U # this is an NxM array
    Sig2_a -= tf.reduce_sum((A@W)*A, axis=1)

    # get the moderation factor. Implicit in here is approximating the logistic sigmoid with
    # a probit by setting the probit and sigmoid slopes to be equal at the origin. This is where
    # the factor of pi/8 comes from.
    if np.sum(Sig2_a.numpy()>0) != Sig2_a.shape[0]:
        print("Sig2_a: ", Sig2_a)
        print("min(Sig2_a): ", min(Sig2_a.numpy()))
        assert False
    kappa_sig2_a = (1. + np.pi * Sig2_a / 8.)**(-1./2)

    # calculate the moderated argument of the logit
    a = mu_a * kappa_sig2_a
    return -tf.math.softplus(-a)

def bayes_logistic_prob(X, mu_beta, sigma):
    """ Posterior predictive logistic regression probability.  Uses probit approximation
        to the logistic regression sigmoid. Also has overflow prevention via exponent truncation.

    adapted from: https://github.com/Valassis-Digital-Media/bayes_logistic

    Parameters
    ----------
    X : array-like, shape (N, p)
        array of covariates
    mu_beta : array-like, shape (p, )
        array of fitted MAP parameters
    sigma : covariance matrix of laplace approximation, array-like, shape (p, p) or (p, )

    Returns
    -------
    log predictive probability of y=1

    References
    ----------
    Chapter 8 of Murphy, K. 'Machine Learning a Probabilistic Perspective', MIT Press (2012)
    Chapter 4, Section 5.2 of Bishop, C. 'Pattern Recognition and Machine Learning', Springer (2006)

    """
    # unmoderated argument of exponent
    mu_a = np.dot(X, mu_beta)

    # find  the moderation
    if len(sigma.shape) == 2:
        sigma_a = tf.reduce_sum( (X@sigma)  * X, axis=1)
    elif len(sigma.shape) == 1:
        sigma_a = np.sum(X * (sigma * X), axis=1)
    else:
        raise ValueError(' You must either use the full Hessian or its diagonal as a vector')

    # get the moderation factor. Implicit in here is approximating the logistic sigmoid with
    # a probit by setting the probit and sigmoid slopes to be equal at the origin. This is where
    # the factor of pi/8 comes from.
    kappa_sig2_a = (1. + np.pi * sigma_a / 8.)**(-1./2.)

    # calculate the moderated argument of the logit
    a = mu_a * kappa_sig2_a

    return -tf.math.softplus(-a)

def logistic(a):
    a = np.clip(a, -20,20)
    return 1./(1. + np.exp(-a))

def phi(y, a):
  return -tf.math.softplus(-y*a)

# Obtain a function that returns 1st order gradients.
dphi_da_full = tfe.gradients_function(phi, params=[1])
dphi_da = lambda y, a: dphi_da_full(y, a)[0]

d2phi_da2_full = tfe.gradients_function(dphi_da, params=[1])
d2phi_da2 = lambda y, a: d2phi_da2_full(y, a)[0]


def generate_data(N, D, mu_X, sigma_X, beta, sigma_obs=None, linear=False):
    """generate_data generates data from a Bayesian logistic regression or
    linear regression

    Args:
        N: number of datapoints
        D: dimensionality of datapoints
        mu_X: prior mean of data
        sigma_X: prior covariance of data
        beta: paramter
        sigma_obs: observation std if linear regression
        linear: set true if linear regression

    Returns:
        X, Y (on scale of -1, 1)
    """
    # Sample covariates (shape [N, D] )
    if len(sigma_X.shape)==1:
        X = np.random.normal(mu_X,sigma_X,size=[N,D])
    else:
        X = np.random.multivariate_normal(mu_X,sigma_X,size=[N])

    # ensure sigma_obs is included iff linear regression
    if linear:
        assert sigma_obs is not None
    else:
        assert sigma_obs is None

    # sample responses
    if linear:
        mu = X.dot(beta)
        Y = np.random.normal(loc=mu, scale=sigma_obs)
    else:
        mu = logistic(X.dot(beta))
        Y = np.random.binomial(1, mu)
        Y = 2.*Y-1.
    return X, Y

def gen_all_data_and_splits(N, D, variance_decay_rate, mu_beta=None,
        sigma_beta=None, rotate=False, verbose=False, max_variance=5., test_size=5000,
        include_bias=False,beta=None, min_variance=1e-5, sigma_obs=None,
        linear=False):
    """gen_all_data_and_splits generates all synthetic data for one trial.


    This includes parameters, training, test and out of sample data (i.e. data with
        covariates drawn from a different distribution).


    Args:
        N: number of train samples
        D: dimension of data
        variance_decay_rate: rate of geometric decay in the variance of
            successive dimensions (>1).
        mu_beta: prior mean
        sigma_beta: prior variance
        include_bias: set true to include a bias by constraining each
           x_{i,1}=1
        beta: value of parameter (sampled from prior if None)
        sigma_obs: observation std if linear regression
        linear: set true if linear regression

    Returns:
        returns a train test and out of sample data as well as the parameter
        used to generate the data.

    """

    # we want a test of 1000 samples for every test
    N_total = N + test_size
    variances = np.array([min_variance + max_variance*variance_decay_rate**-i for i in
        range(D)])

    # set distribution over data
    mu_X = np.zeros(D)
    if rotate:
        sigma_X = np.diag(variances)
        a = np.random.random(size=(D, D))
        rotation, _ = np.linalg.qr(a)
        mu_X = np.dot(rotation, mu_X)
        sigma_X = np.dot(rotation, np.dot(sigma_X, rotation.T))
        if include_bias: # alter covariance of data such that X[i, 0] = 0
            sigma_X[:,0] *= 0.
            sigma_X[0, :] *= 0.
            sigma_X*=variance_decay_rate
            mu_X[0]=1.
    else:
        sigma_X = variances
        assert not include_bias # we haven't handled this case yet...

    # sample parameter from prior
    if beta is None:
        if len(sigma_beta.shape) == 1:
            beta = np.random.normal(mu_beta, sigma_beta)
        else:
            beta = np.random.multivariate_normal(mu_beta, sigma_beta, size=[])

    if verbose: print("generating data")
    X, Y = generate_data(N_total, D, mu_X, sigma_X, beta=beta,
            sigma_obs=sigma_obs, linear=linear)

    # generate out of sample data
    if verbose: print("generating oos data")
    if rotate:
        sigma_X_oos = np.diag(variances)
        a = np.random.random(size=(D, D))
        rotation, _ = np.linalg.qr(a)
        mu_X = np.dot(rotation, mu_X)
        sigma_X_oos = np.dot(rotation, np.dot(sigma_X_oos, rotation.T))
        if include_bias: # alter covariance of data such that X[i, 0] = 0
            sigma_X_oos[:,0] *= 0.
            sigma_X_oos[0, :] *= 0.
            sigma_X_oos*=variance_decay_rate
            mu_X[0]=1.
    else:
        sigma_X_oos = np.array(variances)[np.random.permutation(D)]
        assert not include_bias # we haven't handled this case yet...
    X_oos, Y_oos = generate_data(test_size, D, mu_X, sigma_X_oos,beta=beta,
            sigma_obs=sigma_obs, linear=linear)

    # Make train/test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=N, )
    data = [
            ("Train", tf.constant(X_train), tf.constant(Y_train)),
            ("Test", tf.constant(X_test), tf.constant(Y_test)),
            ("Out of Sample", tf.constant(X_oos), tf.constant(Y_oos)),
            ("beta", tf.constant(beta))
            ]
    return data

def plot_contours(ax, mu, cov, color='black', label=None, stds=[1,2,3]):
    """plot_contours plots contours of a Gaussian

    These roughly equate to +-1, 2 and 3 st-devs from the mean

    The distance of a contour from the mean is SD/(v^T \Sigma^{-1} v)^{\frac{1}{2}})
    """
    prec = np.linalg.inv(cov)
    for std in stds:
        pts =[]
        for theta in np.linspace(0,2*np.pi, 200):
            v=np.array([np.cos(theta),np.sin(theta)])
            l = std/np.sqrt(v.dot(prec).dot(v))
            pts.append(mu+v*l)
        pts = np.array(pts)
        ax.plot(pts[:,0], pts[:,1], label=label if std==stds[0] else None, c=color)

def steepest_descent_line_search(obj, jac, D, max_iter=250, prec=1e-3,
        alpha=1., verbose=False, log_iter=10, x0=None):
    """steepest_descent_line_search finds the optimal beta by steepest
    descent with a line search

    Args:
        obj: objective
        jac: jacobian
        D: dimensionality
        max_iter: maximum number of iterations
        prec: precision at which to stop (when within precision of the
            minimum).
        alpha: strong convexity parameter
    """
    beta = tf.zeros(D) if x0 is None else x0 # initial beta
    i=0 # iteration
    ls_decrease = 0.5 # line search backtracking decrease rate
    rho_init = ls_decrease
    while i<max_iter:
        grad = jac(beta)
        grad_norm = np.linalg.norm(grad)
        max_dist = grad_norm/alpha
        if verbose:
            if i%log_iter==0:
                print("%04d:\tobj=%0.03f\n\tmax_dist: %0.02f\n\tbeta[:5]: %s"%(i,obj(beta), max_dist, beta.numpy()[:5]))
        if max_dist < prec: break
        descent_dir = grad / grad_norm

        # Perform backtracking line search
        rho=rho_init
        obj_ls = obj(beta-grad/alpha)
        ls_steps = 0
        while True:
            back_step_obj = obj(beta-rho*grad/alpha)
            if back_step_obj > obj_ls or grad_norm*rho/alpha < prec/10.:
                if np.linalg.norm(jac(beta-rho*grad/alpha)) < grad_norm:
                    rho_init = rho/(ls_decrease**2)
                    break
            ls_steps += 1
            obj_ls = back_step_obj
            rho *= ls_decrease

        # update beta
        beta = beta - rho*grad/alpha
        i+=1 # increment iteration

    if verbose or i==max_iter: print("exited after %d iterations within %0.02f of "
            "minimum"%(i, max_dist))

    return beta


## Style dictionary

### Create style dictionary for plots
def style_dict(Ms, D):
    # Ms < D
    Ms = [M for M in Ms if M <= D]
    style_components = ["label_names", "colors", "markers"]
    style_dict = {style_component:{} for style_component in style_components}

    style_dict['colors']["ADVI_MF"] = 'y'
    style_dict['colors']["ADVI_FR"] = '#ff9875'
    style_dict['colors']["Diagonal_Laplace"] = 'r'
    style_dict['colors']["Prior"] = 'm'
    style_dict['colors']["HMC"] = 'c'
    style_dict['colors']["NUTS"] = 'c'
    style_dict['colors']["Laplace"] = 'k'
    style_dict['colors']["Fast_Laplace"] = 'b'
    style_dict['colors']["Random_Laplace"] = 'c'
    style_dict['colors']["Fast_HMC"] = 'g'
    style_dict['colors']["Fast_NUTS"] = 'g'
    #for M, c in zip(Ms, sns.color_palette("Blues",n_colors=len(Ms))):
    for M, c in zip(Ms, sns.cubehelix_palette(n_colors=len(Ms)+1, rot=-.4)):
        style_dict['colors']["Fast_Laplace_(M=%d)"%M] = c
    for M, c in zip(Ms, sns.color_palette("Purples",n_colors=len(Ms))):
        style_dict['colors']["Random_Laplace_(M=%d)"%M] = c
    for M, c in zip(Ms, sns.color_palette("Blues",n_colors=len(Ms))):
        style_dict['colors']["Fast_HMC_(M=%d)"%M] = c
        style_dict['colors']["Fast_NUTS_(M=%d)"%M] = c

    # Next marker type
    for approx in style_dict['colors']: # iterate over methods
        style_dict['markers'][approx]='s' # square
        style_dict["label_names"][approx] = approx.replace("_"," ")

    for M in Ms:
        style_dict["label_names"]["Fast_Laplace_(M=%d)"%M]="LR-Laplace (M=%d)"%M
        style_dict["label_names"]["Fast_NUTS_(M=%d)"%M]="LR-NUTS (M=%d)"%M

    style_dict["label_names"]["Fast_Laplace"]="LR-Laplace"

    # Font-size
    style_dict["title_fontsize"]=11
    #style_dict["axis_fontsize"]=7.5
    #style_dict["legend_fontsize"]=6
    #style_dict["tick_fontsize"]=6
    style_dict["axis_fontsize"]=10
    style_dict["legend_fontsize"]=7.5
    style_dict["tick_fontsize"]=7.5
    #style_dict["axis_fontsize"]=8
    #style_dict["legend_fontsize"]=6.5
    #style_dict["tick_fontsize"]=6.5
    style_dict["linewidth"]=1
    style_dict['marker_size_single'] = 8
    style_dict['marker_size_series'] = 4


    return style_dict
