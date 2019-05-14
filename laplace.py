import tensorflow as tf
import utils
from scipy import optimize
from scipy import stats
import sklearn
import numpy as np
ftol = 1e-6

def MAP(X, Y, Sigma_beta, phi, dphi_da, verbose=False, max_iter=250,
        x0=None, ftol=ftol):
    """MAP finds the maximum a posteriori (MAP) estimate of the parameter

    Args:
        X: design [N, D]
        Y: responses in {-1, 1} [N]
        Sigma_beta: prior variance [D]
        phi: glm mapping function
        dphi_da: 1st derivative
        verbose: set true to log optimization progress
    """
    activations = lambda beta: tf.einsum('ND,D->N',X, beta)

    ### Define the log likelihood and its jacobian
    # construct log likelihood
    ll = lambda beta: tf.reduce_sum(phi(Y, activations(beta)))

    # construct Jacobian
    jac_l = lambda beta: tf.reduce_sum(dphi_da(Y,
        activations(beta))[:, None]*X, axis=0)

    ### Define the log prior, and its jacobian and Hessian
    lp = lambda beta: -0.5*tf.reduce_sum(tf.log(2*np.pi*Sigma_beta)) \
         -0.5*tf.reduce_sum(beta**2/Sigma_beta)
    jac_p = lambda beta: -beta*(1./Sigma_beta)

    ### Define log posterior, and its jacobian and Hessian
    l_post =    lambda beta: ll(beta)     + lp(beta)
    jac_post =  lambda beta: jac_l(beta)  + jac_p(beta)

    ### Optimize for MAP (projected space)
    x0 = np.zeros(X.shape[1]) if x0 is None else x0
    if True:
        res = optimize.minimize(
                lambda beta: np.float64(-l_post(tf.constant(beta))),
                x0=x0, method="L-BFGS-B",
                jac=lambda beta: np.array(-jac_post(tf.constant(beta))),
                options={"ftol":ftol, "gtol":1e-6})
        if verbose: print("MAP: opt result: ", res)
        beta_map = tf.constant(res.x)
    elif False:
        res = optimize.minimize(
                lambda beta: np.float64(-l_post(tf.constant(beta, dtype=tf.float32))),
                x0=x0, method="L-BFGS-B",
                jac=lambda beta: np.array(-jac_post(tf.constant(beta,
                    dtype=tf.float32)),
                    dtype=np.float64), options={"ftol":ftol, "gtol":1e-6})
        if verbose: print("MAP: opt result: ", res)
        beta_map = tf.constant(res.x, dtype=tf.float32)
    else:
        x0 = tf.cast(x0, dtype=tf.float32)
        beta_map = utils.steepest_descent_line_search(lambda b: -l_post(b),
                lambda b: -jac_post(b),
                X.shape[1], verbose=True, max_iter=max_iter, x0=x0)
        if verbose: print("MAP: opt result--- beta=", beta_map,
                "f=",-l_post(beta_map))
    return beta_map

def diag_laplace(X, Y, Sigma_beta, phi, dphi_da, d2phi_da2, beta_map=None,
        ftol=ftol):
    """diag_laplace performs a diagonal laplace approximation over parameters

    First optimisation (if MAP not provided) then inversion.

    We only allow the case where the prior is zero mean isotropic Gaussian.

    Args:
        X: design [N, D]
        Y: responses in {-1, 1} [N]
        Sigma_beta: prior variance [D]
        phi: glm mapping function
        dphi_da: 1st derivative
        d2phi_da2: 2nd derivative
        beta_map: maximum a posteriori beta

    Returns:
        mu_N, Sigma_N  (of shapes [D] and [D] )
    """
    activations = lambda beta: tf.einsum('ND,D->N',X, beta)
    ### Define the log likelihood, and its jacobian and Hessian
    # construct log likelihood
    ll = lambda beta: tf.reduce_sum(phi(Y, activations(beta)))

    # construct Jacobian
    jac_l = lambda beta: tf.reduce_sum(dphi_da(Y,
        activations(beta))[:, None]*X, axis=0)

    # construct diagonal of Hessian
    hess_l_diag = lambda beta: tf.reduce_sum(d2phi_da2(Y,
        activations(beta))[:,None]*(X**2), axis=0)

    ### Define the log prior, and its jacobian and Hessian
    lp = lambda beta: -0.5*tf.reduce_sum(tf.log(2*np.pi*Sigma_beta)) \
         -0.5*tf.reduce_sum(beta**2/Sigma_beta)
    jac_p = lambda beta: -beta*(1./Sigma_beta)
    hess_p_diag = lambda beta: -1./Sigma_beta

    ### Define log posterior, and its jacobian and Hessian
    l_post =    lambda beta: ll(beta)     + lp(beta)
    jac_post =  lambda beta: jac_l(beta)  + jac_p(beta)
    hess_post_diag = lambda beta: hess_l_diag(beta) + hess_p_diag(beta)

    ### Optimize for approximate MAP (projected space)
    if beta_map is None:
        res = optimize.minimize(
                lambda beta: np.float64(-l_post(tf.constant(beta, dtype=tf.float32))),
                np.zeros(X.shape[1]), method="L-BFGS-B",
                jac=lambda beta: np.array(-jac_post(tf.constant(beta,
                    dtype=tf.float32)),
                    dtype=np.float64), options={"ftol":ftol, "gtol":1e-6})
        if verbose: print("opt result: ", res)
        beta_map = tf.constant(res.x, dtype=tf.float32)

    ### Form Gaussian approximation with diagonal covariance (This simpler
    ### because the mean is zero mean & Isotropic)
    mu_n = beta_map
    Sigma_N = 1./(-hess_post_diag(beta_map))
    return mu_n, Sigma_N

def laplace(X, Y, Sigma_beta, phi, dphi_da, d2phi_da2, beta_map=None,
        ftol=ftol):
    """laplace performs a full laplace approximation over parameters

    First optimisation (if MAP not provided) then inversion.

    We only allow the case where the prior is zero mean isotropic Gaussian.

    Args:
        X: design [N, D]
        Y: responses in {-1, 1} [N]
        Sigma_beta: prior variance [D]
        phi: glm mapping function
        dphi_da: 1st derivative
        d2phi_da2: 2nd derivative
        beta_map: maximum a posteriori beta

    Returns:
        mu_N, Sigma_N
    """
    activations = lambda beta: tf.einsum('ND,D->N', X, beta)
    ### Define the log likelihood, and its jacobian and Hessian
    ll = lambda beta: tf.reduce_sum(phi(Y, activations(beta)))
    jac_l = lambda beta: tf.reduce_sum(dphi_da(Y,
        activations(beta))[:, None]*X, axis=0)
    hess_l = lambda beta: tf.transpose(X)*d2phi_da2(Y,activations(beta))[None] @X

    ### Define the log prior, and its jacobian and Hessian
    lp = lambda beta: -0.5*tf.reduce_sum(tf.log(2*np.pi*Sigma_beta)) \
         -0.5*tf.reduce_sum(beta**2/Sigma_beta)
    jac_p = lambda beta: -beta*(1./Sigma_beta)
    hess_p = lambda beta: -tf.diag(1./Sigma_beta)

    ### Define log posterior, and its jacobian and Hessian
    l_post =    lambda beta: ll(beta)     + lp(beta)
    jac_post =  lambda beta: jac_l(beta)  + jac_p(beta)
    hess_post = lambda beta: hess_l(beta) + hess_p(beta)

    ### Optimize for approximate MAP (projected space)
    if beta_map is None:
        res = optimize.minimize(
                lambda beta: np.float64(-l_post(tf.constant(beta, dtype=tf.float32))),
                np.zeros(X.shape[1]), method="L-BFGS-B",
                jac=lambda beta: np.array(-jac_post(tf.constant(beta,
                    dtype=tf.float32)),
                    dtype=np.float64), options={"ftol":ftol, "gtol":1e-6})
        if verbose: print("opt result: ", res)
        beta_map = tf.constant(res.x, dtype=tf.float32)

    ### Form Gaussian approximation (This simpler because the mean is zero
    ### mean & Isotropic)
    mu_n = beta_map
    Sigma_N = tf.linalg.inv(-hess_post(mu_n))
    return mu_n, Sigma_N

def fast_laplace(XU, U, Y, Sigma_beta, phi, dphi_da, d2phi_da2,
        verbose=False, ftol=ftol):
    """fast_laplace performs a laplace approximation using a low rank data
    approximation

    We only allow the case where the prior is zero mean isotropic Gaussian.

    This first finds the approximate MAP, beta_map, and then forms a Laplace
    approximation using the approximate likelihood.

    Defining the hessian of the log likelihood w.r.t U^T\\beta as:
        H = \\nabla_{U^T\\beta}^2 log p(Y|XUU^T\\beta)

    Sigma_n^{-1} ~=  Sigma_beta^{-1} - U^T H U
    mu_n = U UT_beta_map

    To avoid an explicit DxD inversion, use woodbury matrix lemma as
    Sigma_n = (Sigma_beta^-1 - U H U^T)^-1
    = Sigma_beta - (Sigma_beta U) (U^T Sigma_beta U -H^{-1})^-1 (U^T Sigma_beta)
    = Sigma_beta - (Sigma_beta U) W (U^T Sigma_beta),
    where W = (U^T Sigma_beta U - H^{-1})^-1

    When the Hessian of the log likelihood is nearly low rank, this
    computation can become very numerically unstable.  To handle these
    issues we take several steps to ensure stability.
       - We add a small nugget to H before inverting
       - We enforce symmetry in H before inverting to ensure eigenvalues are
         real.
       - We enforce symmettry of W^{-1} before inverting.

    Args:
        XU: projected data [N by M]
        U: projection matrix with orthonormal columns, so U^TU=I [D by M]
        Y: responses in {-1, 1} [N]
        Sigma_beta: prior variance [D]
        phi: GLM mapping function, maps (x_i^T\\beta, y_i) --> log p(y_i|x_i, y)
        dphi_da: 1st derivative
        d2phi_da2: 2nd derivative

    Returns:
        mu_N, W (aproximate posterior mean and woodbury_factor)
    """
    activations = lambda UT_beta: tf.einsum('NM,M->N', XU, UT_beta)

    ### Define the log likelihood, and its jacobian and Hessian
    ll = lambda UT_beta: tf.reduce_sum(phi(Y, activations(UT_beta)))
    jac_l = lambda UT_beta: tf.reduce_sum(dphi_da(Y,
        activations(UT_beta))[:, None]*XU, axis=0)

    ### Define the log prior, and its jacobian and Hessian
    UT_Sigma_beta_U = tf.matmul(tf.transpose(U), Sigma_beta[:,None]*U)
    UT_Sigma_beta_U_inv = tf.linalg.inv(UT_Sigma_beta_U)

    lp = lambda UT_beta: -0.5*tf.linalg.logdet(2*np.pi*UT_Sigma_beta_U) \
         -0.5*tf.einsum('i,i->',
                 tf.einsum('i,ij->j', UT_beta, UT_Sigma_beta_U_inv),
                 UT_beta)
    jac_p = lambda UT_beta: -tf.einsum('m,mk->k', UT_beta,
            UT_Sigma_beta_U_inv)

    ### Define log posterior, and its jacobian and Hessian
    l_post = lambda UT_beta: ll(UT_beta) + lp(UT_beta)
    jac_post = lambda UT_beta: jac_l(UT_beta) + jac_p(UT_beta)

    ### Optimize for approximate MAP (projected space)
    if True:
        res = optimize.minimize(
                lambda beta: np.float64(-l_post(tf.constant(beta))),
                np.zeros(XU.shape[1]), method="L-BFGS-B",
                jac=lambda beta: np.array(-jac_post(tf.constant(beta))),
                options={"ftol":ftol, "gtol":1e-6})
        UT_beta_map = tf.constant(res.x)
    else:
        res = optimize.minimize(
                lambda beta: np.float64(-l_post(tf.constant(beta, dtype=tf.float32))),
                np.zeros(XU.shape[1]), method="L-BFGS-B",
                jac=lambda beta: np.array(-jac_post(tf.constant(beta,
                    dtype=tf.float32)),
                    dtype=np.float64), options={"ftol":ftol, "gtol":1e-6})
        UT_beta_map = tf.constant(res.x, dtype=tf.float32)
    if verbose: print("Fast Laplace opt result: ", res)

    ### Form Gaussian approximation (This is simple because the mean is zero
    ### mean & isotropic)
    mu_n = tf.einsum('DM,M->D', U, UT_beta_map)

    # Cast all to higher precision to avoid instability
    U = tf.cast(U, dtype=tf.float64); XU = tf.cast(XU, dtype=tf.float64)
    Sigma_beta = tf.cast(Sigma_beta, dtype=tf.float64)
    Y=tf.cast(Y, dtype=tf.float64)
    UT_beta_map = tf.cast(UT_beta_map, dtype=tf.float64)

    # W = (U^T Sigma_beta U - H^{-1})^-1
    # second derivative of mapping function
    phipp = d2phi_da2(Y, tf.einsum('NM,M->N', XU, UT_beta_map))[None]
    hess_l_val = tf.matmul(tf.transpose(XU)*phipp, XU)

    # Enforce symmetry to ensure eigenvalues are real, and add noise for
    # positivity of eigenvals.  Neglecting this step procudes numerical
    # intability issues.
    hess_l_val -= tf.eye(int(XU.shape[1]), dtype=tf.float64)*1e-5
    hess_l_val += tf.transpose(hess_l_val); hess_l_val /= 2.
    if verbose:
        eig_vals_hess_l = np.linalg.eigvals(hess_l_val.numpy())
        print("hess_l --- min/max:", min(eig_vals_hess_l), max(eig_vals_hess_l))
    W_inv = tf.einsum('DM,DK->MK', U, Sigma_beta[:,None]*U) - tf.linalg.inv(hess_l_val)
    W_inv += tf.transpose(W_inv); W_inv /= 2.
    if verbose:
        eig_vals_w_inv = np.linalg.eigvals(W_inv.numpy())
        print("W_inv --- min/max:", min(eig_vals_w_inv), max(eig_vals_w_inv))
    W = tf.linalg.inv(W_inv)
    # Enforce symmetry to ensure eigenvalues are real, and add noise for
    # positivity of eigenvals
    W += tf.transpose(W); W /= 2.

    eig_vals = np.linalg.eigvals(W.numpy())
    if verbose: print("W --- min/max:", min(eig_vals), max(eig_vals))
    if not min(eig_vals)>0.:
        print("W", W,"\neig_vals:", eig_vals, "\nmin(eig_vals):", min(eig_vals))
        assert False
    return mu_n, W
