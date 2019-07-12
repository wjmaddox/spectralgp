import math
import torch
import matplotlib.pyplot as plt
import sys
#import seaborn as sns
import copy
import numpy as np



def plot_kernel(alt_sampler, data_mod, gen_kern):
    last_samples = min(10, alt_sampler.gsampled[0].size(1))
    with torch.no_grad():
        # preprocess the spectral samples #
        data_mod.eval()
        
        for dim in range(len(alt_sampler.gsampled)):
            tau = torch.linspace(0, 3, 300)
            plt_kernels = torch.zeros(tau.nelement(), last_samples)
            for ii in range(last_samples):
                out_samples = alt_sampler.gsampled[dim][0, :, -last_samples:]
                data_mod.covar_module.set_latent_params(out_samples[:,ii], idx=dim)
                plt_kernels[:, ii] = data_mod.covar_module.kernels[dim](tau, torch.zeros(1,1)).squeeze(1)

            plt_kernels = plt_kernels.detach().cpu().numpy()
            tau = tau.cpu().numpy()
            colors = ["#eac100", "#5893d4", "#10316b", "#070d59"]
            plt.plot(tau, plt_kernels[:, 0], color=colors[1], alpha=0.5,
                    label="Sampled Kernels")
            plt.plot(tau, plt_kernels, color=colors[1], alpha=0.5)

            plt.ylabel("K(tau)", fontsize=14)
            plt.xlabel("tau", fontsize=14)
            plt.title("Kernel Reconstruction; Dimension: {}".format(dim), fontsize=20)
            plt.legend(loc=1)
            plt.grid(alpha = 0.5)
            plt.show()

# def plot_kernel_2(alt_sampler, omega, data_mod, latent_mod, gen_kern, n_to_plt=5):
#     tau_min = 0
#     tau_max = 3
#     tau = torch.linspace(tau_min, tau_max)
#     last_samples = min(20, alt_sampler.gsampled[0].shape[1])
#     out_samples = alt_sampler.gsampled[0][0, :, -last_samples:].t()
#     post_for_ci = utils.get_kernels(out_samples, tau, data_mod).detach().numpy()
#     lower_kern = np.percentile(post_for_ci,2.5, 0)
#     upper_kern = np.percentile(post_for_ci, 97.5, 0)

#     last_samples = min(n_to_plt, alt_sampler.gsampled[0].shape[1])
#     out_samples = alt_sampler.gsampled[0][0, :, -last_samples:].t()
#     post_kerns = utils.get_kernels(out_samples, tau, data_mod).detach()
#     true_kern = gen_kern(tau, torch.zeros(1,1)).squeeze(1)

#     post_mean = data_mod.covar_module.latent_mean(omega).detach()
#     post_mean = utils.get_kernels(post_mean.unsqueeze(0), tau, data_mod).detach()

#     sns.set_palette("muted")
#     sns.set_style("whitegrid")
#     sns.set_context("poster")
#     sns.despine()
#     plt.figure(figsize=(10,9))
#     plt.plot(tau.numpy(), post_kerns.t().numpy())
#     plt.plot(tau.numpy(), post_mean.t().numpy(), color="steelblue",
#              linewidth=2.5, linestyle='--', label='Posterior mean')
#     plt.plot(tau.numpy(), true_kern.detach().numpy(), label="Truth",
#             linewidth=2.5, linestyle="--")
#     plt.fill_between(tau.numpy(), lower_kern, upper_kern,
#              color="steelblue", alpha=0.1, label=r'$\pm 2$ SD')
#     plt.ylim((-1, 3.5))
#     plt.xlim((tau_min, tau_max))
#     plt.title("Posterior Kernels", fontsize=28)
#     plt.xlabel("Tau", fontsize=28)
#     plt.ylabel("Correlation", fontsize=28)
#     plt.legend()
#     plt.show()
