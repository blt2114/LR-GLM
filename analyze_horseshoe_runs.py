import utils
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import matplotlib
import seaborn as sns

sns.set_style('white')
matplotlib.rcParams['legend.frameon'] = False
matplotlib.rcParams['figure.dpi'] = 750
matplotlib.rcParams['savefig.dpi'] = 750
matplotlib.rcParams['xtick.major.pad']='2'
matplotlib.rcParams['ytick.major.pad']='2'
matplotlib.rcParams['axes.labelpad']='0.5'

plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif')


def load_relevant(path):
    with open(path, "rb") as f:
        data_dict = pickle.load(f)
        beta = data_dict['beta']
        time = data_dict['time']
        summary = data_dict['summary']
    return beta, time, summary

if __name__ == "__main__":

    N = 1000
    D = 200

    beta_full, time_full, summary_full = load_relevant('../../saved_models/full_rank_N_{0}_D_{1}.pkl'.format(N, D))
    sd_beta_full = summary_full[:, 2]
    np.std(beta_full, axis=0)
    avg_beta_full = summary_full[:, 0]
    avg_rhat_full = np.mean(summary_full[:, 9])
    avg_ess_full = np.mean(summary_full[:, 8])
    M_grid = np.arange(20, np.min([D, N]), D // 10)
    style_dict = utils.style_dict(M_grid, 200)
    times_M = []
    error_M = []
    r_hats_M = []
    ess_M = []
    sd_error_M = []
    for M in M_grid:
      beta_low, time_low, summary_low = load_relevant('../../saved_models/low_rank_N_{0}_D_{1}_M_{2}.pkl'.format(N, D, M))
      sd_beta_low = summary_low[:, 2]
      times_M.append(time_low)
      avg_beta_low = summary_low[:, 0]
      avg_rhat_low = np.mean(summary_low[:, 9])
      avg_ess_low = np.mean(summary_low[:, 8])
      rel_error_beta = np.sum((avg_beta_full - avg_beta_low) ** 2 / np.sum(avg_beta_full ** 2))
      rel_error_sd = np.sum((sd_beta_full - sd_beta_low) ** 2 / np.sum(sd_beta_full ** 2))
      error_M.append(rel_error_beta)
      r_hats_M.append(avg_rhat_low)
      ess_M.append(avg_ess_low)
      sd_error_M.append(rel_error_sd)


    # FINAL ICML PLOTS

    # Posterior mean error plot
    figsize=(2.5,2)
    sns.set_style('white')
    sns.set_context('notebook', rc={'lines.linewidth': style_dict["linewidth"]})
    plt.figure(figsize=figsize, dpi=750)
    plt.plot(times_M, error_M)
    plt.axvline(time_full, label='Full Rank', color='red')
    plt.xlabel('Time (s)', size=style_dict["axis_fontsize"])
    plt.ylabel('Post. Mean Rel. Error',  size=style_dict["axis_fontsize"])
    plt.yscale('log')
    sns.despine()
    plt.tight_layout()
    plt.savefig('horseshoe_means.png')
    plt.close()

    # Posterior sd error plot
    sns.set_style('white')
    sns.set_context('notebook', rc={'lines.linewidth': style_dict["linewidth"]})
    plt.figure(figsize=figsize , dpi=750)
    plt.plot(times_M, sd_error_M)
    plt.axvline(time_full, label='Full Rank', color='red')
    plt.xlabel('Time (s)', size=style_dict['axis_fontsize'])
    plt.ylabel('Post. SDs Rel. Error',  size=style_dict["axis_fontsize"])
    plt.yscale('log')
    sns.despine()
    plt.tight_layout()
    plt.savefig('horseshoe_sds.png')
    plt.close()
