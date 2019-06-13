import math
import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import spectralgp
from spectralgp.kernels.recover_spectral_density import get_spectral_density_from_kernel
from bnse.bnse import bse

from spectralgp.samplers import AlternatingSampler
from spectralgp.models import ExactGPModel, SpectralModel
import matplotlib.patches as mpatches
from custom_plotting import plot_predictions_real_dat, plot_spectrum, plot_kernel, plot_data_para_models

from spectralgp.sampling_factories import ss_factory, ess_factory

import data
from fit_parametric_model import fit_parametric_model
import spectralgp.utils as utils
import argparse

import sys

plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
torch.set_default_dtype(torch.float64)
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='(str) options: "airline"...',
                        default='airline', type=str,
                        choices=['airline', 'jura', 'QP', 'audio1', 'RBF', 'co2', 'sinc', 'wind',
                                'gas', 'yacht', '3droad', 'electric', 'protein', 'video', 'elevators',
                                'linear', 'SM', 'hr1', 'hr2', 'sunspots'])
    parser.add_argument('--nx', help='(int) number of data points for simulated data',
                        default=100, type=int)
    parser.add_argument('--period', help='(float) period for QP kernel',
                        default=1.7, type=float)
    parser.add_argument('--lengthscale', help='(float) lengthscale for sim kernel',
                        default=2., type=float)
    parser.add_argument('--spacing', help='(str) should data be evenly spaced or randomly sampled',
                        default='even', type=str, choices=['even', 'random'])
    parser.add_argument('--noise', help='(bool) should generated data be generated with noise',
                        default='False', type=bool)
    parser.add_argument('--optim_iters', help='(int) number of optimization iterations',
                        default=1, type=int)
    parser.add_argument('--omega_max', help='(float) maximum value of omega', default=1., type=float)
    parser.add_argument('--save', help='(bool) save model output (samples and model and omega)?',
                        default=False, type=bool)
    parser.add_argument('--ci', help='(bool) plot with CIs?',
                        default=False, type=bool)
    return parser.parse_args()
def main(argv, seed=88):
    '''
    runs ESS with fixed hyperparameters:
    run with -h for CL arguments description
    '''
    # parse CL arguments #
    torch.random.manual_seed(seed)
    ##########################################
    ## some set up and initialization stuff ##
    ##########################################
    ## options ##
    args = parse()
    nx = args.nx
    dataset = args.data
    gen_pars = [args.lengthscale, args.period]
    fpath = "./saved_outputs/" + dataset

    out_samples = torch.load(fpath + "_samples.pt")
    omega = torch.load(fpath + "_omega.pt")

    train_x, train_y, test_x, test_y, gen_kern = data.read_data(dataset, nx=nx,
                                                                spacing = args.spacing,
                                                                gen_pars = gen_pars)
    # plt.plot(test_x.numpy(), test_y.numpy())
    # plt.show()
    data_lh = gpytorch.likelihoods.GaussianLikelihood(noise_prior=gpytorch.priors.SmoothedBoxPrior(1e-8, 1e-3))
    data_mod = spectralgp.models.SpectralModel(train_x, train_y, data_lh, omega=omega, normalize = False,
                            symmetrize = False, transform=utils.link_function)

    data_mod.load_state_dict(torch.load(fpath + "_model.pt"))

    ## set the good pars from previous training ##
    rbf_mod = fit_parametric_model(train_x, train_y, iters=100, kernel='rbf')
    matern_mod = fit_parametric_model(train_x, train_y, iters=100, kernel='matern')
    sm_mod = fit_parametric_model(train_x, train_y, iters=5000, kernel='sm')

    ## BNSE Model ##
    full_x = torch.cat((train_x, test_x)).sort()[0]

    my_bse = bse(train_x.numpy(), train_y.numpy(), full_x.numpy())
    my_bse.train()
    my_bse.compute_moments()

    last_samples = out_samples.shape[1]
    pred_data = torch.zeros(len(full_x), last_samples)
    lower_pred = torch.zeros(len(full_x), last_samples)
    upper_pred = torch.zeros(len(full_x), last_samples)

    print(list(data_mod.named_parameters()))
    # plt.plot(omega.numpy(), out_samples.numpy())
    # plt.show()

    data_mod.eval()
    with torch.no_grad():
        for ii in range(last_samples):
                data_mod.covar_module.latent_params = out_samples[:,ii]
                data_mod.set_train_data(train_x, train_y)
                data_mod.eval()

                out = data_mod(full_x)
                lower_pred[:, ii], upper_pred[:, ii] = out.confidence_region()
                pred_data[:, ii] = out.mean

    colors = cm.get_cmap("tab10")
    plt.figure(figsize=(20, 9))
    fkl, = plt.plot(full_x.cpu().numpy(), pred_data[:, 0].detach().cpu().numpy(), color=colors(0), alpha = 0.5,
            label="FKL")
    plt.plot(full_x.cpu().numpy(), pred_data.detach().cpu().numpy(), color=colors(0), alpha = 0.5)
    # print(args.ci)
    ci_patch = mpatches.Patch(color=colors(0), alpha=0.1, label=r'$\pm 2$' + " SD")
    if args.ci:
        for ii in range(last_samples):
                plt.fill_between(full_x.cpu().numpy(), lower_pred[:, ii].detach().cpu().numpy(),
                            upper_pred[:, ii].detach().cpu().numpy(), color=colors(0), alpha = 0.03)
    train_data, = plt.plot(train_x.cpu().numpy(), train_y.cpu().numpy(), linestyle='None', marker='.',
            color=colors(1), label="Training Data", markersize=16)
    if test_y is not None:
        test_data, = plt.plot(test_x.cpu().numpy(), test_y.cpu().numpy(), color=colors(1),
                label="Test Data", linewidth=3.)


    ## plot the simple models ##
    lw = 2.5
    rbf_pred = rbf_mod(full_x).mean.detach().cpu().numpy()
    matern_pred = matern_mod(full_x).mean.detach().cpu().numpy()
    sm_pred = sm_mod(full_x).mean.detach().cpu().numpy()
    rbf, = plt.plot(full_x.cpu().numpy(), rbf_pred, color=colors(2), label="RBF",
            linewidth=lw)
    matern, = plt.plot(full_x.cpu().numpy(), matern_pred, color=colors(6), label="Matern",
            linewidth=lw)
    sm, = plt.plot(full_x.cpu().numpy(), sm_pred, color=colors(3), label="SM", linestyle=(0, (5, 5)),
            linewidth=lw)
    bnse, = plt.plot(full_x.cpu().numpy(), my_bse.post_mean, color=colors(5), label="BNSE",
            linewidth=lw)

    plt.rcParams['legend.fontsize'] = 18
    if dataset=='airline':
        plt.ylabel('Passengers (std.)', fontsize=24)
        plt.xlabel('Time (std.)', fontsize=24)
    else:
        plt.ylabel('Y', fontsize=24)
        plt.xlabel('X', fontsize=24)
    plt.title("Quasi-Periodic Data", fontsize=28)
    # plt.ylim((-5, 5))
    # handles, labels = plt.get_legend_handles_labels()
    plt.legend(handles=[fkl, ci_patch, rbf, matern, sm, bnse, train_data, test_data], loc=2)
    plt.grid(alpha = 0.5)
    plt.show()

    plt.rcParams['legend.fontsize'] = 22
    if gen_kern is not None:

        with torch.no_grad():
            # preprocess the spectral samples #
            data_mod.eval()

            tau = torch.linspace(0, 3, 300)
            plt_kernels = torch.zeros(tau.nelement(), last_samples)
            for ii in range(last_samples):
                    data_mod.covar_module.latent_params = out_samples[:,ii]
                    plt_kernels[:, ii] = data_mod.covar_module(tau, torch.zeros(1,1)).squeeze(1)
            if gen_kern is not None:
                    true_kernel = gen_kern(tau, torch.zeros(1,1)).squeeze(1)

            # prior_kern = data_mod.covar_module(tau, torch.zeros(1,1)).squeeze(1)

            plt_kernels = plt_kernels.detach().cpu().numpy()
            tau = tau.cpu().numpy()
            # colors = ["#eac100", "#5893d4", "#10316b", "#070d59"]
            plt.figure(figsize=(10, 9))
            plt.plot(tau, plt_kernels[:, 0], color=colors(0), alpha=0.5,
                    label="Sampled Kernels")
            plt.plot(tau, plt_kernels, color=colors(0), alpha=0.5)
            # plt.plot(tau, prior_kern.detach().cpu().numpy(), color=colors[2], linestyle=":",
            #         label="Prior Mean")

            plt.plot(tau, true_kernel.detach().cpu().numpy(), color=colors(1),
                    label="Truth", linewidth=2.5)

            plt.ylabel("K(Tau)", fontsize=24)
            plt.xlabel("Tau", fontsize=24)
            plt.title("Kernel Reconstruction", fontsize=28)
            plt.legend(loc=1)
            plt.grid(alpha = 0.5)
            plt.show()

        with torch.no_grad():

            # preprocess the spectral samples #
            spectral_mean = data_mod.covar_module.latent_mean(omega).detach()
            out_samples = out_samples + spectral_mean.unsqueeze(1)

            spectrum_samples = utils.link_function(out_samples)
            spectrum_samples = spectrum_samples.detach().cpu().numpy()

            # true spectrum
            tau = torch.linspace(0, 4, 400)
            true_kernel = gen_kern(tau, torch.zeros(1,1)).squeeze(1)
            true_spect, true_omg = get_spectral_density_from_kernel(gen_kern, locs=omega.nelement(), s_range=max(omega))
            true_spect = true_spect.detach().cpu().numpy()

            omega = omega.cpu().numpy()

            # colors = ["#eac100", "#5893d4", "#10316b", "#070d59"]
            plt.figure(figsize=(10, 9))
            plt.plot(omega, spectrum_samples[:, 0], alpha=0.5, color=colors(0),
                    label="Posterior Samples")
            plt.plot(omega, spectrum_samples[:,1:], alpha=0.5, color=colors(0))
            plt.plot(true_omg.cpu().numpy(), true_spect, color=colors(1),
                    label="True Spectrum", linewidth=2.5)


            plt.xlabel("Frequency", fontsize=24)
            m1 = np.max(spectrum_samples)
            m2 = np.max(true_spect)
            plt.ylim(bottom=0, top=1.2*np.max((m1, m2)))
            plt.ylabel("Spectral Density", fontsize=24)
            plt.title("Spectral Density", fontsize=28)
            plt.legend(loc=1)
            plt.grid(alpha = 0.5)
            plt.show()




if __name__ == '__main__':
    main(sys.argv[1:])
