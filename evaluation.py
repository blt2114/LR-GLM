import utils
import ipdb
import posterior
import pickle
import tensorflow as tf
import numpy as np
from scipy import stats
import os
import matplotlib.pyplot as plt

from matplotlib import cm

def calibration_curve(Y, logistic_p, cutoffs, min_count_in_bin=2):
    """calibration_curve creates an empirical calibration curve for the
    predictive probabilities and lables provided

    the calibration curve contains for each of a number of confidence intervals
    the observed fraction of positive samples observed in that interval.

    Args:
        Y: observed labels [N]
        logistic_p: the predicticted probabilies [N]
        cutoffs: the cutoffs for each interval
        min_count_in_bin: to 'somewhat' reduce sampling noise, set fraction
            to None if fewer than 2 samples in the given bin.

    Returns:
        calibration curve (list of observed fractions)
    """
    cal_curve = []
    bin_size = cutoffs[1]-cutoffs[0]
    for cutoff in cutoffs:
        bin_idxs = tf.where(tf.logical_and(logistic_p > cutoff, logistic_p <= cutoff+bin_size))
        frac_pos_in_bin = tf.reduce_mean(
                tf.cast(tf.equal(tf.gather(Y,bin_idxs), 1), tf.float64))
        n_in_bin = bin_idxs.shape[0]
        if n_in_bin>=min_count_in_bin:
            rate=frac_pos_in_bin.numpy()
        else:
            rate=None
        cal_curve.append(rate)
    return cal_curve

def plot_samples_gaussian_approx(X, Y, mu_beta, sigma_beta, eps, plot_lim,
        x_2_boundary, x1_grid, base_fn, title=None, ax_label_size=26,
        tick_fontsize=17, has_bias=True):
    f, axarr = plt.subplots(ncols=2, figsize=[11,5.5])
    if has_bias:
        mu_beta_plot, sigma_beta_plot  = mu_beta[1:], sigma_beta[1:, 1:]
    else:
        mu_beta_plot, sigma_beta_plot  = mu_beta, sigma_beta
    utils.plot_contours(axarr[1], mu_beta_plot,  cov=sigma_beta_plot, color='red')
    axarr[1].set_xlabel(r"$\beta_1$", fontsize=ax_label_size)
    axarr[1].set_ylabel(r"$\beta_2$", fontsize=ax_label_size)
    b_plot_lim = 3.5
    axarr[1].set_xlim([-b_plot_lim, b_plot_lim]); axarr[1].set_ylim([-b_plot_lim, b_plot_lim])
    for tick in axarr[1].yaxis.get_major_ticks(): tick.label.set_fontsize(tick_fontsize)
    for tick in axarr[1].xaxis.get_major_ticks(): tick.label.set_fontsize(tick_fontsize)

    # plot data-points and decision boundaries
    axarr[0].scatter(X[:, 0], X[:, 1], c=np.float32(Y), s=140.,
            cmap=cm.get_cmap("binary"), edgecolors="k", label="Datapoints",
            linewidths=2)
    chol =  np.linalg.cholesky(sigma_beta)
    print("plotting samples decisions")
    for i, eps_i in enumerate(eps):
        beta_i = mu_beta + chol.dot(eps_i)
        x_2_boundary_vals = x_2_boundary(beta_i, x1_grid)
        axarr[0].plot(x1_grid, x_2_boundary_vals, label=title, c='k',
                linewidth=0.5)
    axarr[0].set_xlim([-plot_lim, plot_lim]); axarr[0].set_ylim([-plot_lim, plot_lim])
    axarr[0].set_xlabel(r"$X_1$", fontsize=ax_label_size)
    axarr[0].set_ylabel(r"$X_2$", fontsize=ax_label_size)
    for tick in axarr[0].yaxis.get_major_ticks(): tick.label.set_fontsize(tick_fontsize)
    for tick in axarr[0].xaxis.get_major_ticks(): tick.label.set_fontsize(tick_fontsize)
    plt.suptitle(title, fontsize=25)
    plt.tight_layout(pad=2.5)
    plt.savefig(base_fn+title+"_2D_vis.png")
    plt.clf()

def plot2D_posterior_samples(
    X, Y, beta_map, mu_beta, sigma_beta, diag_mu_N, diag_Sigma_N, full_mu_N, full_Sigma_N,
    fast_mu_N, fast_Sigma_N, full_beta_samples=None, fast_beta_samples=None,
    base_fn=None, has_bias=True):
    """plot2D_posterior_samples plots points and 2-class decision boundaries
    for logistic regression.

    Args:
        X, Y: features and responses
        beta_map: maximum a posteriori beta
        has_bias: true if first column of X is 1s and beta[0] is a bias
    """
    print("Plotting")
    # for plotting, convert to numpy arrays
    if type(X) != np.ndarray:
        X = X.numpy()
        Y = Y.numpy()
    if has_bias:
        X = X[:, 1:]

    # for 2D
    n_samples = 20
    plot_lim = 10
    x1_grid=np.linspace(-plot_lim, plot_lim, 100)
    if has_bias:
        x_2_boundary = lambda beta, x1: -(x1*beta[1]+ beta[0])/beta[2]
    else:
        x_2_boundary = lambda beta, x1: -x1*beta[0]/beta[1]
    idxs_1 = np.where(Y==1.)[0]
    idxs_0 = np.where(Y!=1.)[0]

    colors = ['k']
    x_2_boundary_vals_map = x_2_boundary(beta_map, x1_grid)

    all_eps = [np.random.normal(size=[3 if has_bias else 2]) for _ in range(n_samples)]

    print('plotting approximations')

    # prior
    print('\tprior')
    plot_samples_gaussian_approx(X, Y, mu_beta, np.diag(sigma_beta),
            all_eps, plot_lim, x_2_boundary, x1_grid, base_fn, "Prior",
            has_bias=has_bias)

    # diagonal Laplace
    print('\tdiagonal')
    plot_samples_gaussian_approx(X, Y, diag_mu_N.numpy(), np.diag(diag_Sigma_N),
            all_eps, plot_lim, x_2_boundary, x1_grid, base_fn,
            "Diagonal Laplace", has_bias=has_bias)

    # full Laplace
    print('\tfull')
    plot_samples_gaussian_approx(X, Y, full_mu_N.numpy(), full_Sigma_N.numpy(),
            all_eps, plot_lim, x_2_boundary, x1_grid, base_fn,
            "Full Laplace", has_bias=has_bias)

    # fast Laplace
    print('\tfast again')
    plot_samples_gaussian_approx(X, Y, fast_mu_N.numpy(), fast_Sigma_N.numpy(),
            all_eps, plot_lim, x_2_boundary, x1_grid, base_fn,
            "LR-Laplace", has_bias=has_bias)

    # plot samples from full HMC
    if full_beta_samples is not None:
        plt.scatter(X[:, 0], X[:, 1], c=np.float32(Y),
                cmap=cm.get_cmap("binary"), edgecolors="k",
                label="Datapoints")
        x_2_boundary_vals_map = x_2_boundary(beta_map, x1_grid)
        plt.plot(x1_grid, x_2_boundary_vals_map, label="MAP", c='k')

        for i in np.random.permutation(int(full_beta_samples.shape[0]))[:n_samples]:
            beta_i = full_beta_samples[i]
            x_2_boundary_vals = x_2_boundary(beta_i, x1_grid)
            plt.plot(x1_grid, x_2_boundary_vals, label="HMC", c=colors[i%len(colors)])
        plt.xlim([-plot_lim, plot_lim]); plt.ylim([-plot_lim, plot_lim])
        plt.title("HMC")
        plt.show()

    # plot samples from fast HMC
    if fast_beta_samples is not None:
        plt.scatter(X[:, 0], X[:, 1], c=np.float32(Y),
                cmap=cm.get_cmap("binary"), edgecolors="k", label="Datapoints")
        x_2_boundary_vals_map = x_2_boundary(beta_map, x1_grid)
        plt.plot(x1_grid, x_2_boundary_vals_map, label="MAP", c='k')

        for i in np.random.permutation(int(fast_beta_samples.shape[0]))[:n_samples]:
            beta_i = fast_beta_samples[i]
            x_2_boundary_vals = x_2_boundary(beta_i, x1_grid)
            plt.plot(x1_grid, x_2_boundary_vals, label="fast HMC", c=colors[i%len(colors)])
        plt.xlim([-plot_lim, plot_lim])
        plt.ylim([-plot_lim, plot_lim])
        plt.title("Fast HMC")
        plt.show()

def evaluate_credible_set_calibration(results_fn, approx, data, n_bins=20):
    """evaluate_credible_set_calibration

    Args:
        results_fn: name of file to write output to (should be .pkl)
        posetrior: a Posterior object to evaluate
        data: Training, test and out of sample sets.  a list of tuples of
            ("name", X, Y)
        n_bins: number of bins in calibration curve

    """
    # crash if trying to write results to already existent file
    # TODO put back in
    #assert not os.path.isfile(results_fn)
    _, beta = data[-1]
    credible_percentiles = approx.credible_percentiles(beta.numpy())
    cal_curve = approx.credible_set_calibration(credible_percentiles)

    f=open(results_fn, 'wb')
    pickle.dump(cal_curve, f)
    f.close()

def plot_credible_set_calibration(plot_fn, base_fn, approximations, rep,
        style_dict):

    # First collect all calibration curves
    approx_calcurves = []
    for approx in approximations:
        fn = base_fn+"_"+approx+"_rep=%d_credset_calibration.pkl"%rep
        f = open(fn, 'rb')
        cal_curve = pickle.load(f)
        f.close()
        approx_calcurves.append((approx, cal_curve))


    # Make plot of calibration
    plt.figure(figsize=[10,10.])
    for approx, cal_curve in approx_calcurves:
        label_name = style_dict["label_names"][approx]
        color = style_dict["colors"][approx]
        x = np.linspace(0, 1, len(cal_curve)+1) # define x for plotting
        plt.plot(x, [0]+list(cal_curve), label=label_name, c=color)

    tick_fontsize = style_dict['tick_fontsize']
    for tick in plt.gca().yaxis.get_major_ticks(): tick.label.set_fontsize(tick_fontsize)
    for tick in plt.gca().xaxis.get_major_ticks(): tick.label.set_fontsize(tick_fontsize)

    # plot diagonal for reference
    #plt.plot([0,1],[0,1],c='#929591', linewidth=5, linestyle='dashed')
    plt.plot([0,1],[0,1],c='k', linewidth=4, linestyle='dashed')

    plt.legend(loc="lower center", bbox_to_anchor=(.45, -0.41),
            ncol=3, fontsize=style_dict['legend_fontsize']*0.8,
            frameon=True)
    plt.tight_layout(rect=(0.10, 0.26, 0.9, 0.95))
    #plt.title("Credible Set Calibration")
    plt.xlabel("Credible set Percentile", fontsize=style_dict['axis_fontsize'])
    plt.ylabel("Fraction of Paramters in Credible Set", fontsize=style_dict['axis_fontsize'])

    plt.savefig(plot_fn)


def evaluate_prediction(results_fn, posterior, data, include_oos=False,
        include_bayes=False, logistic=True, include_calibration=False,
        bin_size=0.05):
    """evaluate_prediction

    Args:
        results_fn: name of file to write output to (should be .pkl)
        posetrior: a Posterior object to evaluate
        data: Training, test and out of sample sets.  a list of tuples of
            ("name", X, Y)
        include_bayes: set true to report predictions with Bayesian
            uncertainty in addition to predictions from the point estimate.

    """

    # crash if trying to write results to already existent file
    # TODO put back in
    #assert not os.path.isfile(results_fn)

    # for now only handle logistic regression
    assert logistic

    results = {}
    for (name, X, Y) in data[:3 if include_oos else 2]:
        for bayes in [False] if not include_bayes else [False, True]:
            if bayes: name+= "_bayes"
            results[name] = {}

            # Make predictions
            Y_pred_log_prob = posterior.predict(X, bayes=bayes)
            Y_pred_log_prob_not = tf.log(1. - tf.exp(
                tf.clip_by_value(Y_pred_log_prob, clip_value_max=-1e-7,
                    clip_value_min=-1e8)))

            # Compute mean log likelihood
            log_prob_by_sample = Y_pred_log_prob*tf.cast(tf.equal(Y, 1.),
                        dtype=tf.float64)+ \
                    Y_pred_log_prob_not*tf.cast(tf.equal(Y, -1.),
                        dtype=tf.float64)
            log_prob_mean = tf.reduce_mean(log_prob_by_sample)
            results[name]["log_likelihood"] = log_prob_mean.numpy()

            # Compute accuracy
            correct_by_sample = tf.cast(tf.equal(
                (Y+1.)/2.,
                tf.cast(Y_pred_log_prob>tf.cast(tf.log(0.5), dtype=tf.float64), dtype=tf.float64)
                ), dtype=tf.float64)
            acc = tf.reduce_mean(correct_by_sample)
            results[name]["accuracy"] = acc.numpy()

            # Compute calibration
            if include_calibration:
                cutoffs = np.arange(0., 1., bin_size)
                cal_curve = calibration_curve(Y, tf.exp(Y_pred_log_prob), cutoffs)
                results[name]["calibration"] = cal_curve

    f=open(results_fn, 'wb')
    pickle.dump(results, f)
    f.close()
    return results

def plot_marginals(baseline_method, results, base_fn, n_reps, style_dict,
        n_params_plot, approximations):
    print("Evaluating marginals")
    # Sample subset of parameter indices to compare marginals
    D = results[0][baseline_method]['mean'].shape[0]
    for rep in range(n_reps):
        param_idxs = np.random.permutation(D)[:n_params_plot]
        baseline_mean = results[rep][baseline_method]['mean'][param_idxs]
        baseline_sds = results[rep][baseline_method]['marginal_sds'][param_idxs]
        baseline_label_name = style_dict["label_names"][baseline_method]

        # compare means
        f, axarr = plt.subplots(ncols=2,nrows=1, figsize=[16.,9.])
        for approx in approximations:
            mean = results[rep][approx]['mean'][param_idxs]
            marginal_sds = results[rep][approx]['marginal_sds'][param_idxs]

            label_name = style_dict["label_names"][approx]
            color = style_dict["colors"][approx]
            marker = style_dict["markers"][approx]

            axarr[0].scatter(baseline_mean, mean, label=label_name,
                    marker=marker, edgecolors='k', facecolors=color, s=70)
            axarr[0].set_xlabel("Exact Posterior Mean",
                    fontsize=style_dict['axis_fontsize'])
            axarr[0].set_ylabel("Approximate Posterior Means",
                    fontsize=style_dict['axis_fontsize'])

            axarr[1].scatter(baseline_sds, marginal_sds, label=label_name,
                    marker=marker, edgecolors='k', facecolors=color, s=70)
            axarr[1].set_xlabel("Exact Posterior St-Devs",
                    fontsize=style_dict['axis_fontsize'])
            axarr[1].set_ylabel("Approximate Posterior St-Devs",
                    fontsize=style_dict['axis_fontsize'])

        tick_fontsize = style_dict['tick_fontsize']
        for tick in axarr[1].yaxis.get_major_ticks(): tick.label.set_fontsize(tick_fontsize)
        for tick in axarr[0].yaxis.get_major_ticks(): tick.label.set_fontsize(tick_fontsize)
        for tick in axarr[1].xaxis.get_major_ticks(): tick.label.set_fontsize(tick_fontsize)
        for tick in axarr[0].xaxis.get_major_ticks(): tick.label.set_fontsize(tick_fontsize)

        # for scientific notation on exponent of ticks
        axarr[0].ticklabel_format(style='sci')
        axarr[1].ticklabel_format(style='sci')

        # set ranges for comparison plots
        min_mean_x, max_mean_x= min(baseline_mean)*1.1, max(baseline_mean)*1.1
        min_mean_y, max_mean_y = min_mean_x*1.05, max_mean_x*1.05
        min_sd_x, max_sd_x= min(baseline_sds), max(baseline_sds)
        gap_baseline_sds = max_sd_x - min_sd_x
        min_sd_x -= gap_baseline_sds*0.09; max_sd_x += gap_baseline_sds*0.09
        #min_sd_y = min_sd_x - gap_baseline_sds*0.1; max_sd_y = max_sd_x + gap_baseline_sds*0.1
        min_sd_y = -0.05; max_sd_y = 1.06
        axarr[0].plot([min_mean_x,max_mean_x],[min_mean_x,max_mean_x],c='#929591')
        axarr[0].set_xlim([min_mean_x, max_mean_x]);
        axarr[0].set_ylim([min_mean_y, max_mean_y])
        axarr[1].plot([min_sd_y,max_sd_y],[min_sd_y,max_sd_y],c='#929591')
        axarr[1].set_xlim([min_sd_x, max_sd_x]); axarr[1].set_ylim([min_sd_y, max_sd_y])
        print("min, max x sd: ", min_sd_x, max_sd_x)

        # Save plot as png
        f.tight_layout(pad=8.4, rect=(0, 0.1, 1, 1))
        leg = axarr[0].legend(loc="lower left", bbox_to_anchor=(0.,
            -0.47), ncol=3, fontsize=style_dict['legend_fontsize'],
            frameon=True)
        leg.get_frame().set_edgecolor('k')
        leg.get_frame().set_linewidth(1.0)
        save_fn = base_fn+"_marginals_baseline="+baseline_method+"_rep=%d.png"%rep
        plt.savefig(save_fn)
        plt.close()

def plot_mean_and_cov_error(error_by_approx, base_fn, baseline_method, style_dict,
        plot_cov_spectral_error=False, baseline_mean=None, plot_relative_mean_error=True):
    """plot_mean_and_cov_error plots the norms of the errors in the mean and
    covariance relative to the exact posterior.
    """
    f, axarr = plt.subplots(ncols=2, nrows=1, figsize=[18, 8.])
    for approx, approx_err in error_by_approx.items():
        # Error in means
        axarr[0].errorbar(
                approx_err['time'][:,0], approx_err['mu'][:,0]/(
                    np.linalg.norm(baseline_mean) if
                    plot_relative_mean_error else 1.)
                    ,
                xerr=approx_err['time'][:,1], yerr=approx_err['mu'][:,1],
                marker=style_dict["markers"][approx],
                label=style_dict["label_names"][approx],
                color=style_dict["colors"][approx])
        axarr[0].set_xlabel("Time (s)", fontsize=style_dict['axis_fontsize'])
        axarr[0].set_ylabel(r"$\|\mu_N - \tilde \mu_N \|_2$" +("$/ \|\mu_N\|$" if plot_relative_mean_error else ""),
                fontsize=style_dict['axis_fontsize'])
        axarr[0].set_title("Error in Posterior Mean", fontsize=style_dict['title_fontsize'])

        # Error in covariances
        axarr[1].errorbar(
                approx_err['time'][:,0], approx_err['cov'][:,0],
                xerr=approx_err['time'][:,1], yerr=approx_err['cov'][:,1],
                marker=style_dict["markers"][approx],
                label=style_dict["label_names"][approx],
                color=style_dict["colors"][approx])
        axarr[1].set_xlabel("Time (s)", fontsize=style_dict['axis_fontsize'])
        if plot_cov_spectral_error:
            axarr[1].set_ylabel(r"$\|\Sigma_N - \tilde \Sigma_N \|_2$", fontsize=style_dict['axis_fontsize'])
        else:
            axarr[1].set_ylabel(r"$\| \mathrm{diag}(\tilde \Sigma_N - \Sigma_N) \|_2$", fontsize=style_dict['axis_fontsize'])
        axarr[1].set_title("Error in Covariance", fontsize=style_dict['title_fontsize'])
        if True:
            axarr[1].set_xscale('log'); axarr[1].set_yscale('log')
            axarr[0].set_xscale('log'); axarr[0].set_yscale('log')

        tick_fontsize = style_dict['tick_fontsize']
        for tick in axarr[1].yaxis.get_major_ticks(): tick.label.set_fontsize(tick_fontsize)
        for tick in axarr[0].yaxis.get_major_ticks(): tick.label.set_fontsize(tick_fontsize)
        for tick in axarr[1].xaxis.get_major_ticks(): tick.label.set_fontsize(tick_fontsize)
        for tick in axarr[0].xaxis.get_major_ticks(): tick.label.set_fontsize(tick_fontsize)

    # Save plot as png
    f.tight_layout(pad=7.4)
    leg = axarr[0].legend(loc="lower left", bbox_to_anchor=(0.52,
        -0.38), ncol=3, fontsize=style_dict['legend_fontsize'],
        frameon=True)
    leg.get_frame().set_edgecolor('k')
    leg.get_frame().set_linewidth(1.0)
    save_fn = base_fn+"_means_and_covariances_error_baseline="+baseline_method+".png"
    plt.savefig(save_fn)
    plt.close()

def evaluate_posterior(base_fn, baseline_method, approximations, style_dict,
        n_reps, plot_cov_spectral_error=True, n_params_plot=20):
    """evaluate_posterior evaluate posterior evaluate posterior
    approximation methods on recovering the exact posterior means and
    covariances relative to the given baseline.

    This method generates two (sets) of plots.  The first visualizes quality of
    the posterior approximations in term of error of means and covariances
    of parameters, in L2 norm and spectral norm respectively.  For this, the
    average error across replicates is plotted with error in terms of SEMs.

    The second plots a subset of recovered functionals relative to the their
    values under the exact posterior.  Namely the erior expectations and
    marginal variances of parameters under the posterior.  This is done for
    each rep separately.


    Both plots are saved to disc as pngs.


    Args:
        base_fn: base of the paths to posterior pickles. All paths are of
            the form -  base_fn + '_' + method + '_' + rep + ".pkl" .
        baseline_method: string name of method to compare to as baseline
        approximations: list of string names of methods to compare to
        n_reps: number of replicates to evaluate
        plot_cov_spectral_error: Set True to plot covariance error as
            spectral norm, otherwise plot as L2 norm of error in the diagonal
        n_params_plot: i parameter plots, the number of paramters to
            visualize.
        style_dict: dictionary of style of each method to plot (e.g. color/size/maker-type/label name)

    """
    n_sems = 1.
    ### First compute errors and load in all relevant results
    results = {}
    for rep in range(n_reps):
        results[rep]={}
        baseline_fn = utils.method_to_fn(base_fn, baseline_method, rep)

        # load in posterior object taken to be the baseline posterior for
        # comparisons.
        f = open(baseline_fn, 'rb')
        exact_posterior = pickle.load(f)
        f.close()
        if "Fast_Lap" in baseline_method:
            exact_posterior.U = tf.cast(exact_posterior.U, tf.float64)
            exact_posterior.W = tf.cast(exact_posterior.W, tf.float64)

        if plot_cov_spectral_error:
            baseline_mean, baseline_cov = exact_posterior.mean_and_cov()
            results[rep][baseline_method] = {
                    "mean": baseline_mean,
                    "marginal_sds": np.sqrt(np.diag(baseline_cov))
                    }
        else:
            baseline_mean = exact_posterior.mean()
            baseline_sds = exact_posterior.marginal_stdevs()
            print(baseline_method, "baseline_sds.shape", baseline_sds.shape)
            results[rep][baseline_method] = {
                    "mean": baseline_mean,
                    "marginal_sds": baseline_sds
                    }

        print("formulating comparison")
        for approx in approximations:
            approx_fn = utils.method_to_fn(base_fn, approx, rep)
            f = open(approx_fn, 'rb')
            approx_posterior = pickle.load(f)
            f.close()
            if "Fast_Lap" in approx or "Rand" in approx:
                approx_posterior.U = tf.cast(approx_posterior.U, tf.float64)
                approx_posterior.W = tf.cast(approx_posterior.W, tf.float64)
            results[rep][approx]={}
            if plot_cov_spectral_error:
                mean, cov = approx_posterior.mean_and_cov()
                # compute spectral norm
                assert len(cov.shape) == 2
                cov_err = np.linalg.norm(cov-baseline_cov, ord=2)
                results[rep][approx]["cov_err"] = cov_err
                results[rep][approx]["marginal_sds"] = np.sqrt(np.diag(cov))
            else:
                mean = approx_posterior.mean()
                sds = approx_posterior.marginal_stdevs()
                print(approx, "marginal_sds.shape", sds.shape)
                results[rep][approx]["marginal_sds"] = sds
                # Define error in covariances as L2 norm of the error in the
                # diagonal
                cov_err = np.linalg.norm(sds**2-baseline_sds**2)
                results[rep][approx]["cov_err"] = cov_err

            results[rep][approx]["mean"] = mean
            # compute L2 norm of mean error
            mean_err = np.linalg.norm(mean-baseline_mean)
            results[rep][approx]["mean_err"] = mean_err

            results[rep][approx]["time"] = approx_posterior.time

    ### Plot results

    ## First compare marginals
    def plot_all_marginals():
        series_plot = ["Fast_Laplace", "Random_Laplace"]
        print("approximations", approximations)
        if len(list(filter(lambda  s: "NUTS" in s, approximations)))!=0: series_plot.append("Fast_NUTS")
        for series in series_plot:
            series_copy = list(series_plot)
            series_copy.pop(series_copy.index(series))
            approx_copy = list(approximations)
            for series_to_remove in series_copy:
                approx_copy = list(filter(lambda s: not series_to_remove in s,
                    approx_copy))
            plot_marginals(baseline_method, results, base_fn+"_"+series, n_reps, style_dict,
                    n_params_plot, approx_copy)
    plot_all_marginals()

    ## Next compare time against norms of errors means and covariances
    # create a dictionary which will contain, for each approximation the
    # mean and 2*sem of the runtime, error in the mean, and spectral error
    # in the covariance.  Fast-Laplace and Fast-HMC this will have multiple
    # elements.  For the others it will not.
    print("Evaluating norms or errors in means and covariances")
    error_by_approx = {approx:{"time":[], "mu":[], "cov":[]} for approx in
            approximations}

    for approx in error_by_approx:
        mean_errors, cov_errors, times  = [], [], []
        for rep in range(n_reps):
            mean_errors.append(results[rep][approx]['mean_err'])
            cov_errors.append(results[rep][approx]['cov_err'])
            times.append(results[rep][approx]['time'])
        # compute mean and 2xSEM of means, covs and times
        error_by_approx[approx]["time"] = np.array([[np.mean(times), n_sems*stats.sem(times)]])
        error_by_approx[approx]["mu"] = np.array([[np.mean(mean_errors), n_sems*stats.sem(mean_errors)]])
        error_by_approx[approx]["cov"] = np.array([[np.mean(cov_errors), n_sems*stats.sem(cov_errors)]])

    # pull out  fast Laplace approximations
    def combine_into_series(method_base_name, approximations,
            error_by_approx, name_for_series, results):
        """combine_into_series combines all approximations containing method
        base name in their title into a single series

        The individual approximations methods are removed from the
        dictionary of results, 'results_by_approx', and a new entry
        containing the whole series is added in their place.

        """
        series_names = list(filter(lambda name: method_base_name in name,
            approximations))
        # for mean and sem for each value of M
        error_by_approx[name_for_series] = {"time":[], "mu":[], "cov":[]}
        for approx in series_names:
            mean_errors, cov_errors, times  = [], [], []
            for rep in range(n_reps):
                mean_errors.append(results[rep][approx]['mean_err'])
                cov_errors.append(results[rep][approx]['cov_err'])
                times.append(results[rep][approx]['time'])
            # compute mean and 2xSEM of means, covs and times
            error_by_approx[name_for_series]["time"].append([np.mean(times), n_sems*stats.sem(times)])
            error_by_approx[name_for_series]["mu"].append([np.mean(mean_errors), n_sems*stats.sem(mean_errors)])
            error_by_approx[name_for_series]["cov"].append([np.mean(cov_errors), n_sems*stats.sem(cov_errors)])
        error_by_approx[name_for_series]["time"] = np.array(error_by_approx[name_for_series]["time"])
        error_by_approx[name_for_series]["mu"] = np.array(error_by_approx[name_for_series]["mu"])
        error_by_approx[name_for_series]["cov"] = np.array(error_by_approx[name_for_series]["cov"])
        # remove individual entries from the results
        for name in series_names: error_by_approx.pop(name)

    if len(list(filter(lambda name: "Fast_HMC" in name,
        approximations))) != 0:
        combine_into_series("Fast_HMC", approximations, error_by_approx,
                "Fast_HMC", results)
    if len(list(filter(lambda name: "Fast_NUTS" in name,
        approximations))) != 0:
        combine_into_series("Fast_NUTS", approximations, error_by_approx,
                "Fast_NUTS", results)
    if len(list(filter(lambda name: "Fast_Lap" in name,
        approximations))) != 0:
        combine_into_series("Fast_Lap", approximations, error_by_approx,
                "Fast_Laplace", results)
    if len(list(filter(lambda name: "Random" in name,
        approximations))) != 0:
        combine_into_series("Random", approximations, error_by_approx,
                "Random_Laplace", results)

    assert len(list(filter(lambda name: "HMC" in name,approximations))) ==0

    #ipdb.set_trace()
    plot_mean_and_cov_error(error_by_approx, base_fn, baseline_method, style_dict,
        plot_cov_spectral_error=plot_cov_spectral_error, baseline_mean=baseline_mean,
        plot_relative_mean_error=True)

def process_calibration_results(calibration_curves, n_not_nan_cuttoff=3):
    """process_calibration_results takes a list of calibration curves and
    returns a single averaged calibration curve

    Args:
        calibration_curves: a list of lists, where each list is a
            calibration curve.
        n_not_nan_cuttoff: number of required entries in the calibration
            curve for a given bin for it to be considered.

    Returns:
        a list of means and a list of 2*SEM uncertainties in the mean
    """
    cal_curves = []
    for cal_curve in calibration_curves:
        cal_curve = [np.nan if v ==None else np.float(v) for v in cal_curve]
        cal_curve = np.array(cal_curve)
        cal_curves.append(cal_curve)
    cal_curves = np.array(cal_curves)

    # filter out entries for which we don't have enough observations
    means, stderrs = [], []
    # iterate through bins, computing means and sems
    for bin_proportions in cal_curves.T:
        if np.count_nonzero(~np.isnan(bin_proportions))<n_not_nan_cuttoff:
            mean = np.nan
            stderr = np.nan
        else:
            mean = np.nanmean(bin_proportions)
            stderr = 2.*stats.sem(bin_proportions, nan_policy='omit')
        means.append(mean); stderrs.append(stderr)
    return np.array(means), np.array(stderrs)

def plot_calibration(base_fn, approximations, style_dict, n_reps,
        include_oos=True, bayes=True, just_HMC=False, no_HMC=False):
    """plot_calibration creates calibration curves for the
    approximations given saved results.  The plots are saved to disc as pngs.


    Args:
        base_fn: base of the paths to posterior pickles. All paths are of
            the form -  base_fn + '_' + method + '_' + rep + ".pkl" .
        approximations: list of string names of methods to compare to
        n_reps: number of replicates to evaluate
        style_dict: dictionary of style of each method to plot (e.g. color/size/maker-type/label name)
        include_oos: set true to include out of sample calibration as well.
        bayes: set True to evaluate posterior predictive in addition
            to predictions with posterior mean.
    """
    categories = ["Train", "Test"]
    if include_oos: categories += ["Out of Sample"]

    sub_title = ""
    if just_HMC:
        approximations = list(filter(lambda approx: "HMC" in approx or "NUTS" in approx,
                approximations))
        sub_title = "_HMC"
    elif no_HMC:
        approximations = list(filter(lambda approx: not ("HMC" in approx or "NUTS" in approx),
                approximations))
        sub_title = "_noHMC"

    ### First compute errors and load in all relevant results
    cal_curves_by_approx = {} # this will contain tuples of means and 2xSEMs
    for approx in approximations:
        cal_curves = {cat:[] for cat in categories}
        for rep in range(n_reps):
            approx_pred_fn = "%s_%s_rep=%d_predictions.pkl"%(base_fn, approx, rep)
            f = open(approx_pred_fn, 'rb')
            results = pickle.load(f)
            f.close()
            for cat in categories:
                cat_name = cat + "_bayes" if bayes and approx != "MAP" else cat
                cal_curves[cat].append(results[cat_name]['calibration'])

        cal_curves_by_approx[approx]={}
        for cat in categories:
            means, sems2 = process_calibration_results(cal_curves[cat])
            assert len(means) == 20
            cal_curves_by_approx[approx][cat]=(means, sems2)

    ### Now plot the curves
    categories = categories[1:]
    ncols = len(categories)
    bins = np.arange(0,1., 0.05) + 0.025
    elinewidth = .5 # errorbar line width
    print("approximations: ", approximations)
    n_exprs_plot = len(approximations)
    bar_sep = 0.001
    offset_start = -bar_sep*n_exprs_plot/2

    f, axarr = plt.subplots(ncols=ncols, nrows=1, figsize=[8*ncols,10.])
    axarr[0].set_ylabel("Fraction Y=1", fontsize=style_dict['axis_fontsize'])
    for col, cat in enumerate(categories):
        for i, approx in enumerate(approximations):
            # add offset so errorbars are not overlapping
            offset = offset_start + (i*bar_sep)

            means, sems2 = cal_curves_by_approx[approx][cat]
            label_name = style_dict["label_names"][approx]
            color = style_dict["colors"][approx]
            axarr[col].errorbar(bins+offset, means,
                yerr=sems2, color=color, label=label_name,
                elinewidth=elinewidth)

        # Format the axis
        axarr[col].set_title(cat, fontsize=style_dict["title_fontsize"])
        axarr[col].plot([0,1],[0,1],c='#929591')
        axarr[col].set_xlim([-0.1,1.1])
        axarr[col].set_ylim([0.,1.])
        axarr[col].set_xlabel("Predicted Fraction Y=1",
            fontsize=style_dict['axis_fontsize'])

    # rect : tuple (left, bottom, right, top), optional
    plt.tight_layout(pad=6., rect=(0,0.1,1, 1))
    axarr[1].legend(loc="lower center", ncol=3,
            fontsize=style_dict['legend_fontsize'],
            bbox_to_anchor=(-.01, -0.32))

    # format and save the figure
    save_fn = base_fn+"_%s%s_calibration.png"%("bayes_" if bayes else "",
            sub_title)
    plt.savefig(save_fn)
    plt.close()


def plot_error_and_NLL(base_fn, approximations, style_dict, n_reps,
        include_oos=True, bayes=True, just_HMC=False, no_HMC=False):
    """plot_error_and_NLL creates plots of the error and negative log
    likelihood for the approximations provided.

    The plots are saved to disc as pngs.

    Args:
        base_fn: base of the paths to posterior pickles. All paths are of
            the form -  base_fn + '_' + method + '_' + rep + ".pkl" .
        approximations: list of string names of methods to compare to
        n_reps: number of replicates to evaluate
        style_dict: dictionary of style of each method to plot (e.g. color/size/maker-type/label name)
        include_oos: set true to include out of sample calibration as well.
        bayes: set True to evaluate posterior predictive in addition
            to predictions with posterior mean.
    """
    n_sems = 1.
    categories = ["Train", "Test"]
    if include_oos: categories += ["Out of Sample"]

    sub_title = ""
    if just_HMC:
        approximations = list(filter(lambda approx: "HMC" in approx or "NUTS" in approx,
                approximations))
        sub_title = "_HMC"
    elif no_HMC:
        approximations = list(filter(lambda approx: not ("HMC" in approx or "NUTS" in approx),
                approximations))
        sub_title = "_noHMC"

    ### First compute errors and load in all relevant results
    # this will contain means and 2xSEMs
    results_by_approx = {}
    for approx in approximations:
        accs, LLs, times = {cat:[] for cat in categories}, {cat:[] for cat in categories}, []
        for rep in range(n_reps):
            # load in posterior approx for time
            approx_fn = "%s_%s_rep=%d.pkl"%(base_fn, approx, rep)
            f = open(approx_fn, 'rb')
            results = pickle.load(f)
            f.close()
            times.append(results.time)

            # load in results on predictions
            approx_pred_fn = "%s_%s_rep=%d_predictions.pkl"%(base_fn, approx, rep)
            f = open(approx_pred_fn, 'rb')
            results = pickle.load(f)
            f.close()
            for cat in categories:
                cat_name = cat + "_bayes" if bayes and approx != "MAP" else cat
                accs[cat].append(results[cat_name]['accuracy'])
                LLs[cat].append(results[cat_name]['log_likelihood'])

        ## compute and save mean and 2xSEM for all results to plot
        results_by_approx[approx] = {}

        # for time
        time_mean, time_2sem = np.mean(times), n_sems*stats.sem(times)
        results_by_approx[approx]['time'] = [time_mean], [time_2sem]

        # for error and NLL
        for cat in categories:
            results_by_approx[approx][cat] = {}
            errs = 100*(1. - np.array(accs[cat]))
            err_mean, err_2sem = np.mean(errs), n_sems*stats.sem(errs)
            results_by_approx[approx][cat]["err"]= [err_mean], [err_2sem]

            NLLs = -np.array(LLs[cat])
            NLL_mean, NLL_2sem = np.mean(NLLs), n_sems*stats.sem(NLLs)

            results_by_approx[approx][cat]["NLL"] = [NLL_mean], [NLL_2sem]

    def combine_into_series(method_base_name, approximations,
            results_by_approx, name_for_series):
        """combine_into_series combines all approximations containing method
        base name in their title into a single series

        The individual approximations methods are removed from the
        dictionary of results, 'results_by_approx', and a new entry
        containing the whole series is added in their place.

        """
        series_names = list(filter(lambda name: method_base_name in name,
            approximations))
        # for mean and sem for each value of M
        series_results = {cat:{"err":([],[]), "NLL":([],[])} for cat in
                categories}
        series_results['time']=([], [])
        for approx in series_names:
            series_results['time'][0].append(results_by_approx[approx]['time'][0][0])
            series_results['time'][1].append(results_by_approx[approx]['time'][1][0])
            for cat in categories:
                series_results[cat]['err'][0].append(results_by_approx[approx][cat]['err'][0][0])
                series_results[cat]['err'][1].append(results_by_approx[approx][cat]['err'][1][0])
                series_results[cat]['NLL'][0].append(results_by_approx[approx][cat]['NLL'][0][0])
                series_results[cat]['NLL'][1].append(results_by_approx[approx][cat]['NLL'][1][0])

        # remove individual entries from the results
        for name in series_names: results_by_approx.pop(name)

        # add in full series
        results_by_approx[name_for_series] = series_results

    ### Group the approximations with different complexities in to single
    # series.
    if len(list(filter(lambda s: "Fast_Lap" in s, approximations))) !=0:
        combine_into_series("Fast_Lap", approximations,
            results_by_approx, "Fast_Laplace")
    if len(list(filter(lambda s: "Fast_NUTS" in s, approximations))) !=0:
        combine_into_series("Fast_NUTS", approximations,
            results_by_approx, "Fast_NUTS")
    if len(list(filter(lambda s: "Fast_HMC" in s, approximations))) !=0:
        combine_into_series("Fast_HMC", approximations,
            results_by_approx, "Fast_HMC")
    if len(list(filter(lambda s: "Random" in s, approximations))) !=0:
        combine_into_series("Random", approximations,
                results_by_approx, "Random_Laplace")


    ### Now plot the curves
    ncols = len(categories)
    f, axarr = plt.subplots(ncols=ncols, nrows=2, figsize=[6*ncols,9.5])
    axarr[0, 0].set_ylabel("Error (%)", fontsize=style_dict['axis_fontsize'])
    axarr[1, 0].set_ylabel("Mean NLL (nats)", fontsize=style_dict['axis_fontsize'])
    for i, approx in enumerate(results_by_approx):
        time_mean, time_2sem = results_by_approx[approx]['time']
        label_name = style_dict["label_names"][approx]
        color = style_dict["colors"][approx]
        marker=style_dict["markers"][approx]

        for col, cat in enumerate(categories):
            err_mean, err_2sem = results_by_approx[approx][cat]["err"]
            NLL_mean, NLL_2sem = results_by_approx[approx][cat]["NLL"]
            axarr[0, col].errorbar(time_mean, err_mean, yerr=err_2sem,
                    xerr=time_2sem, label=label_name, color=color,
                    marker=marker)

            axarr[1, col].errorbar(time_mean, NLL_mean, yerr=NLL_2sem,
                    xerr=time_2sem, label=label_name, color=color,
                    marker=marker)

    # Set X axis labels and titles
    tick_fontsize = style_dict['tick_fontsize']
    for col, cat in enumerate(categories):
        axarr[1, col].set_xlabel("Time (s)",
        fontsize=style_dict['axis_fontsize'])
        axarr[0, col].set_title(cat, fontsize=style_dict['title_fontsize'])
        for ax in axarr[:, col]:
            for tick in ax.yaxis.get_minor_ticks():
                tick.label.set_fontsize(tick_fontsize)
            for tick in ax.xaxis.get_minor_ticks():
                tick.label.set_fontsize(tick_fontsize)
            for tick in ax.yaxis.get_major_ticks(): tick.label.set_fontsize(tick_fontsize)
            for tick in ax.xaxis.get_major_ticks(): tick.label.set_fontsize(tick_fontsize)
        if False: # set time to log-scale
            axarr[0, col].set_xlim([0.05,max([v['time'][0] for _, v in
                results_by_approx.items()])*1.05])
            axarr[1, col].set_xlim([0.05,max([v['time'][0] for _, v in
                results_by_approx.items()])*1.05])
            axarr[0, col].set_xscale('log')
            axarr[1, col].set_xscale('log')

    # format and save the figure
    save_fn = base_fn+"_%s%sErr_and_NLL.png"%("bayes_" if bayes else "",
            sub_title)
    # rect : tuple (left, bottom, right, top), optional
    plt.tight_layout(pad=5., rect=(0,0.05,1.,1.))
    axarr[1, 1].legend(loc="lower center", ncol=3,
            fontsize=style_dict['legend_fontsize'],
            bbox_to_anchor=(1.17, -0.65))
    plt.savefig(save_fn)
    plt.close()
