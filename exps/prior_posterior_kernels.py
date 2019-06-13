import math
import torch
import gpytorch
import numpy as np
import spectralgp
import matplotlib.pyplot as plt
import seaborn as sns
from spectralgp.samplers import AlternatingSampler
from spectralgp.models import ExactGPModel, SpectralModel
from spectralgp.sampling_factories import ss_factory, ess_factory
import data
from spectralgp.means import LogRBFMean
import spectralgp.utils as utils
import argparse
import sys
torch.set_default_dtype(torch.float64)

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ess_iters', help='(int) number of ess samples per iteration',
                        default=20, type=int)
    parser.add_argument('--mean', help='(str) latent mean, options = "Constant", "LogRBF"',
                        default="LogRBF")
    parser.add_argument('--nomg', help='(int) number of omegas to use',
                        default=100, type=int)
    parser.add_argument('--iters', help='(int) # of ESS iterations to run',
                        default=100, type=int)
    parser.add_argument('--data', help='(str) options: "airline"...',
                        default='airline', type=str,
                        choices=['airline', 'jura', 'QP', 'audio1', 'RBF', 'co2', 'sinc', 'wind',
                                'gas', 'yacht', '3droad', 'electric', 'protein', 'video', 'elevators',
                                'linear', 'SM', 'hr1', 'hr2', 'sunspots'])
    parser.add_argument('--nx', help='(int) number of data points for simulated data',
                        default=400, type=int)
    parser.add_argument('--lengthscale', help='(float) lengthscale for sim kernel',
                        default=2., type=float)
    parser.add_argument('--period', help='(float) period for QP kernel',
                        default=1., type=float)
    parser.add_argument('--noise', help='(bool) should generated data be generated with noise',
                        default='False', type=bool)
    parser.add_argument('--optim_iters', help='(int) number of optimization iterations',
                        default=1, type=int)
    parser.add_argument('--omega_max', help='(float) maximum value of omega', default=1., type=float)
    parser.add_argument('--save', help='(bool) save model output (samples and model and omega)?',
                        default=False, type=bool)
    parser.add_argument('--spacing', help='(str) should data be evenly spaced or randomly sampled',
                        default='even', type=str, choices=['even', 'random'])
    return parser.parse_args()

def main(argv, seed=824):
    args = parse()
    gen_pars = [args.lengthscale, args.period]
    torch.random.manual_seed(seed)

    n_to_plt = 5
    tau_min = 0
    tau_max = 3
    ##########################################
    ## generate data and push to gpu ##
    ##########################################
    ## options ##
    # load data
    train_x, train_y, test_x, test_y, gen_kern = data.read_data(args.data, nx=args.nx, gen_pars=gen_pars,
                                                            spacing='even')
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)
        train_x, train_y, test_x, test_y = train_x.cuda(), train_y.cuda(), test_x.cuda(), test_y.cuda()
        if gen_kern is not None:
            gen_kern = gen_kern.cuda()

    ###########################################
    ## set up the spectral and latent models ##
    ###########################################
    data_lh = gpytorch.likelihoods.GaussianLikelihood(noise_prior=gpytorch.priors.SmoothedBoxPrior(1e-8, 1e-3))
    data_mod = spectralgp.models.SpectralModel(train_x, train_y, data_lh,
        normalize = False, symmetrize = False, num_locs = args.nomg, spacing=args.spacing, omega_max = args.omega_max)
    data_lh.raw_noise = torch.tensor(-3.5)
    omega = data_mod.covar_module.latent_mod.train_inputs[0].squeeze()
    mu_init = data_mod.covar_module.latent_mod.mean_module(omega).detach()


    ################################
    ## set up alternating sampler ##
    ################################

    alt_sampler = spectralgp.samplers.AlternatingSampler([data_mod], [data_lh],
                        spectralgp.sampling_factories.ss_factory,
                        [spectralgp.sampling_factories.ess_factory],
                        totalSamples=args.iters, numInnerSamples=args.ess_iters,
                        numOuterSamples=args.optim_iters)

    alt_sampler.run()

    data_mod.eval()

    #######################
    ## get prior kernels ##
    #######################
    g_init = alt_sampler.gsampled[0][0, :, 0]
    prior_latent_lh = gpytorch.likelihoods.GaussianLikelihood(noise_prior=gpytorch.priors.SmoothedBoxPrior(1e-8, 1e-3))
    prior_latent_mod = spectralgp.models.ExactGPModel(omega, g_init, prior_latent_lh)

    prior_draws = prior_latent_mod(omega).sample(sample_shape=torch.Size((n_to_plt,))).detach()
    to_gen_conf = prior_latent_mod(omega).sample(sample_shape=torch.Size((2000,))).detach()
    prior_draws = prior_draws + mu_init

    tau = torch.linspace(tau_min, tau_max)
    kernels = utils.get_kernels(prior_draws, tau, data_mod, train_x, train_y)

    lu_sampled_kernels = utils.get_kernels(to_gen_conf, tau, data_mod, train_x, train_y)
    lu_sampled_kernels = lu_sampled_kernels.detach().numpy()
    lower_kern = np.percentile(lu_sampled_kernels,2.5, 0)
    upper_kern = np.percentile(lu_sampled_kernels, 97.5, 0)
    mean_kern = utils.get_kernels(mu_init.unsqueeze(0), tau, data_mod, train_x, train_y)

    true_kern = gen_kern(tau, torch.zeros(1,1)).squeeze(1).detach()

    sns.set_palette("muted")
    sns.set_style("whitegrid")
    sns.set_context("poster")
    sns.despine()
    print("mean_kern shape", mean_kern.shape)
    plt.figure(figsize=(10,9))
    plt.plot(tau.numpy(), kernels.t().detach().numpy())
    plt.plot(tau.numpy(), mean_kern.t().detach().numpy(), color="steelblue",
             linewidth=2.5, linestyle='--', label='prior mean')
    plt.fill_between(tau.numpy(), lower_kern, upper_kern,
             color="steelblue", alpha=0.1, label=r'$\pm 2$ SD')
    plt.plot(tau.numpy(), true_kern.numpy(), label="Truth",
            linewidth=2.5, linestyle="--")
    plt.ylim((-1, 3))
    plt.xlim((tau_min, tau_max))
    plt.title("Prior Kernels", fontsize=28)
    plt.xlabel("Tau", fontsize=28)
    plt.ylabel("Correlation", fontsize=28)
    plt.legend()
    plt.show()


    tau = torch.linspace(tau_min, tau_max)
    last_samples = min(100, alt_sampler.gsampled[0].shape[1])
    out_samples = alt_sampler.gsampled[0][0, :, -last_samples:].t()
    post_for_ci = utils.get_kernels(out_samples, tau, data_mod, train_x, train_y).detach().numpy()
    lower_kern = np.percentile(post_for_ci,2.5, 0)
    upper_kern = np.percentile(post_for_ci, 97.5, 0)

    last_samples = min(n_to_plt, alt_sampler.gsampled[0].shape[1])
    out_samples = alt_sampler.gsampled[0][0, :, -last_samples:].t()
    post_kerns = utils.get_kernels(out_samples, tau, data_mod, train_x, train_y).detach()

    post_mean = data_mod.covar_module.latent_mod.mean_module(omega).detach()
    post_mean = utils.get_kernels(post_mean.unsqueeze(0), tau, data_mod, train_x, train_y).detach()

    sns.set_palette("muted")
    sns.set_style("whitegrid")
    sns.set_context("poster")
    sns.despine()
    print("mean_kern shape", mean_kern.shape)
    plt.figure(figsize=(10,9))
    plt.plot(tau.numpy(), post_kerns.t().numpy())
    plt.plot(tau.numpy(), post_mean.t().numpy(), color="steelblue",
             linewidth=2.5, linestyle='--', label='Posterior mean')
    plt.plot(tau.numpy(), true_kern.numpy(), label="Truth",
            linewidth=2.5, linestyle="--")
    plt.fill_between(tau.numpy(), lower_kern, upper_kern,
             color="steelblue", alpha=0.1, label=r'$\pm 2$' + " sd")
    plt.ylim((-0.5, 1.5))
    plt.xlim((tau_min, tau_max))
    plt.xlabel(r'$\tau$')
    plt.ylabel("Correlation")
    plt.legend()
    plt.show()



if __name__ == '__main__':
    main(sys.argv[1:])
