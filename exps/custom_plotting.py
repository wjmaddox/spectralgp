import math
import torch
import matplotlib.pyplot as plt
import sys
import seaborn as sns
import copy
import numpy as np

from spectralgp.kernels.recover_spectral_density import get_spectral_density_from_kernel
import spectralgp.utils as utils

def plot_spectrum(omega, alt_sampler, latent_mod, gen_kern, spectral_mean=None):
    last_samples = 10
    with torch.no_grad():
        # preprocess the spectral samples #
        out_samples = alt_sampler.gsampled[0][0, :, -last_samples:]
        out_samples = out_samples #+ spectral_mean.unsqueeze(1)

        spectrum_samples = torch.exp(out_samples)
        spectrum_samples = spectrum_samples.detach().cpu().numpy()
        # print(spectrum_samples.shape, omega.size())
        # print('area of samples: ', np.trapz(spectrum_samples, omega.squeeze().cpu().numpy(),axis=0))

        # true spectrum
        if gen_kern is not None:
                tau = torch.linspace(0, 4, 400)
                true_kernel = gen_kern(tau, torch.zeros(1,1)).squeeze(1)
                true_spect, true_omg = get_spectral_density_from_kernel(gen_kern, locs=omega.nelement(), s_range=max(omega))
                true_spect = true_spect.detach().cpu().numpy()

    omega = omega.cpu().numpy()

    colors = ["#eac100", "#5893d4", "#10316b", "#070d59"]
    plt.plot(omega, spectrum_samples[:, 0], alpha=0.5, color=colors[1],
            label="Posterior Samples")
    plt.plot(omega, spectrum_samples[:,1:], alpha=0.5, color=colors[1])

    if gen_kern is not None:
        plt.plot(true_omg.cpu().numpy(), true_spect, color=colors[0],
                label="True Spectrum")

    plt.xlabel("Frequency", fontsize=14)
    if gen_kern is not None:
        m1 = np.max(spectrum_samples)
        m2 = np.max(true_spect)
        plt.ylim(bottom=0, top=1.2*np.max((m1, m2)))
    else:
        plt.ylim(bottom=0, top=1.2*np.max(spectrum_samples))
    plt.ylabel("Spectral Density", fontsize=14)
    plt.title("Spectral Density", fontsize=20)
    plt.legend(loc=1)
    plt.grid(alpha = 0.5)
    plt.show()

def plot_predictions_real_dat(alt_sampler, omega, data_mod, latent_mod,train_x, train_y, test_x, test_y):
    last_samples = min(10, alt_sampler.gsampled[0].size(1))
    # preprocess the spectral samples #
    out_samples = alt_sampler.gsampled[0][0, :, -last_samples:].detach()

    full_x = torch.cat((train_x, test_x)).sort()[0]
    pred_data = torch.zeros(len(full_x), last_samples)
    lower_pred = torch.zeros(len(full_x), last_samples)
    upper_pred = torch.zeros(len(full_x), last_samples)

    data_mod.eval()
    with torch.no_grad():
        for ii in range(last_samples):
                data_mod.covar_module.set_latent_params(out_samples[:,ii])
                data_mod.set_train_data(train_x, train_y)
                out = data_mod(full_x)
                lower_pred[:, ii], upper_pred[:, ii] = out.confidence_region()
                pred_data[:, ii] = out.mean

    colors = ["#eac100", "#5893d4", "#10316b", "#070d59"]
    plt.plot(full_x.cpu().numpy(), pred_data[:, 0].detach().cpu().numpy(), color=colors[1], alpha = 0.5,
            label="Prediction")
    plt.plot(full_x.cpu().numpy(), pred_data.detach().cpu().numpy(), color=colors[1], alpha = 0.5)
    plt.plot(train_x.cpu().numpy(), train_y.cpu().numpy(), linestyle='None', marker='.',
            color=colors[0], label="Training Set")
    if test_y is not None:
        plt.plot(test_x.cpu().numpy(), test_y.cpu().numpy(), color=colors[0],
                label="Data")

    for ii in range(last_samples):
        if ii == 0:
            plt.fill_between(full_x.cpu().numpy(), lower_pred[:, ii].detach().cpu().numpy(),
                            upper_pred[:, ii].detach().cpu().numpy(), color=colors[2], alpha = 0.03,
                            label="Confidence Region")
        else:
            plt.fill_between(full_x.cpu().numpy(), lower_pred[:, ii].detach().cpu().numpy(),
                            upper_pred[:, ii].detach().cpu().numpy(), color=colors[2], alpha = 0.03)
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.title("Prediction on Data", fontsize=20)
    plt.legend(loc=1)
    plt.grid(alpha = 0.5)
    plt.show()


def plot_kernel(alt_sampler, omega, data_mod, latent_mod, gen_kern, mu_init):
    last_samples = min(10, alt_sampler.gsampled[0].size(1))

    with torch.no_grad():
        # preprocess the spectral samples #
        out_samples = alt_sampler.gsampled[0][0, :, -last_samples:]
        #out_samples = out_samples #+ utils.get_mean(latent_mod, omega).unsqueeze(1)
        data_mod.eval()

        tau = torch.linspace(0, 3, 300)
        plt_kernels = torch.zeros(tau.nelement(), last_samples)
        for ii in range(last_samples):
                data_mod.covar_module.set_latent_params(out_samples[:,ii])
                data_mod.prediction_strategy = None
                plt_kernels[:, ii] = data_mod.covar_module(tau, torch.zeros(1,1)).squeeze(1)
        if gen_kern is not None:
                true_kernel = gen_kern(tau, torch.zeros(1,1)).squeeze(1)

        #data_mod.covar_module.latent_params = mu_init
        #prior_kern = data_mod.covar_module(tau, torch.zeros(1,1)).squeeze(1)

        plt_kernels = plt_kernels.detach().cpu().numpy()
        tau = tau.cpu().numpy()
        colors = ["#eac100", "#5893d4", "#10316b", "#070d59"]
        plt.plot(tau, plt_kernels[:, 0], color=colors[1], alpha=0.5,
                label="Sampled Kernels")
        plt.plot(tau, plt_kernels, color=colors[1], alpha=0.5)
        #plt.plot(tau, prior_kern.detach().cpu().numpy(), color=colors[2], linestyle=":",
        #        label="Prior Mean")

        if gen_kern is not None:
                plt.plot(tau, true_kernel.detach().cpu().numpy(), color=colors[0],
                        label="Truth")

        plt.ylabel("K(Tau)", fontsize=14)
        plt.xlabel("Tau", fontsize=14)
        plt.title("Kernel Reconstruction", fontsize=20)
        plt.legend(loc=1)
        plt.grid(alpha = 0.5)
        plt.show()

def plot_latent_space(alt_sampler, omega, latent_mod):
    # data stuff #
    last_samples=5
    out_samples = alt_sampler.gsampled[0][0, :, -last_samples:]

    # latent samples #
    with torch.no_grad():
        lat_samples = latent_mod(omega).sample(sample_shape=torch.Size((1000, )))

    # numpy-ify #
    omega = omega.cpu().numpy()
    out_samples = out_samples.detach().cpu().numpy()
    lat_samples = lat_samples.t().cpu().numpy()

    # plotting #
    clrs = ["#eac100", "#0b8457", "#10316b"]
    # post samples
    plt.plot(omega, out_samples[:, 0], color=clrs[1], alpha=0.5,
            label="Posterior Samples")
    plt.plot(omega, out_samples, color=clrs[1], alpha=0.5)
    # latent model samples
    plt.plot(omega, lat_samples[:, 0], color=clrs[2], alpha=0.05,
            label="Latent Model Samples")
    plt.plot(omega, lat_samples, color=clrs[2], alpha=0.05)
    plt.legend(loc=1)
    plt.grid(alpha = 0.5)
    plt.show()

def plot_data_para_models(alt_sampler, omega, data_mod, latent_mod,
                          rbf_mod, matern_mod, sm_mod,
                          train_x, train_y, test_x, test_y):
    last_samples = min(10, alt_sampler.gsampled[0].size(1))
    # preprocess the spectral samples #
    out_samples = alt_sampler.gsampled[0][0, :, -last_samples:].detach()

    full_x = torch.cat((train_x, test_x)).sort()[0]
    pred_data = torch.zeros(len(full_x), last_samples)
    lower_pred = torch.zeros(len(full_x), last_samples)
    upper_pred = torch.zeros(len(full_x), last_samples)

    data_mod.eval()
    with torch.no_grad():
        for ii in range(last_samples):
                data_mod.covar_module.set_latent_params(out_samples[:,ii])
                out = data_mod(full_x)
                lower_pred[:, ii], upper_pred[:, ii] = out.confidence_region()
                pred_data[:, ii] = out.mean

    colors = ["#eac100", "#5893d4", "#10316b", "#070d59"]
    # plt.plot(full_x.cpu().numpy(), pred_data[:, 0].detach().cpu().numpy(), color=colors[1], alpha = 0.5,
    #         label="Prediction")
    plt.plot(full_x.cpu().numpy(), pred_data.detach().cpu().numpy(), color=colors[1], alpha = 0.5)
    plt.plot(train_x.cpu().numpy(), train_y.cpu().numpy(), linestyle='None', marker='.',
            color=colors[0], label="Training Data")
    if test_y is not None:
        plt.plot(test_x.cpu().numpy(), test_y.cpu().numpy(), color=colors[0],
                label="Test Data")

    for ii in range(last_samples):
        plt.fill_between(full_x.cpu().numpy(), lower_pred[:, ii].detach().cpu().numpy(),
                        upper_pred[:, ii].detach().cpu().numpy(), color=colors[2], alpha = 0.03)

    ## plot the simple models ##
    rbf_pred = rbf_mod(full_x).mean.detach().cpu().numpy()
    matern_pred = matern_mod(full_x).mean.detach().cpu().numpy()
    sm_pred = sm_mod(full_x).mean.detach().cpu().numpy()
    plt.plot(full_x.cpu().numpy(), rbf_pred, color='red')
    plt.plot(full_x.cpu().numpy(), matern_pred, color='green')
    plt.plot(full_x.cpu().numpy(), sm_pred, color='orange')

    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.title("Prediction on Data", fontsize=20)
    plt.legend(loc=1)
    plt.grid(alpha = 0.5)
    plt.show()

def plot_kernel_2(alt_sampler, omega, data_mod, latent_mod, gen_kern, n_to_plt=5):
    tau_min = 0
    tau_max = 3
    tau = torch.linspace(tau_min, tau_max)
    last_samples = min(20, alt_sampler.gsampled[0].shape[1])
    out_samples = alt_sampler.gsampled[0][0, :, -last_samples:].t()
    post_for_ci = utils.get_kernels(out_samples, tau, data_mod).detach().numpy()
    lower_kern = np.percentile(post_for_ci,2.5, 0)
    upper_kern = np.percentile(post_for_ci, 97.5, 0)

    last_samples = min(n_to_plt, alt_sampler.gsampled[0].shape[1])
    out_samples = alt_sampler.gsampled[0][0, :, -last_samples:].t()
    post_kerns = utils.get_kernels(out_samples, tau, data_mod).detach()
    true_kern = gen_kern(tau, torch.zeros(1,1)).squeeze(1)

    post_mean = data_mod.covar_module.latent_mean(omega).detach()
    post_mean = utils.get_kernels(post_mean.unsqueeze(0), tau, data_mod).detach()

    sns.set_palette("muted")
    sns.set_style("whitegrid")
    sns.set_context("poster")
    sns.despine()
    plt.figure(figsize=(10,9))
    plt.plot(tau.numpy(), post_kerns.t().numpy())
    plt.plot(tau.numpy(), post_mean.t().numpy(), color="steelblue",
             linewidth=2.5, linestyle='--', label='Posterior mean')
    plt.plot(tau.numpy(), true_kern.detach().numpy(), label="Truth",
            linewidth=2.5, linestyle="--")
    plt.fill_between(tau.numpy(), lower_kern, upper_kern,
             color="steelblue", alpha=0.1, label=r'$\pm 2$ SD')
    plt.ylim((-1, 3.5))
    plt.xlim((tau_min, tau_max))
    plt.title("Posterior Kernels", fontsize=28)
    plt.xlabel("Tau", fontsize=28)
    plt.ylabel("Correlation", fontsize=28)
    plt.legend()
    plt.show()
