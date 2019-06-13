import math
import torch
import gpytorch
import numpy as np

import spectralgp

from spectralgp.samplers import AlternatingSampler
from spectralgp.models import ExactGPModel, SpectralModel

from custom_plotting import plot_predictions_real_dat, plot_spectrum, plot_kernel, plot_data_para_models

from spectralgp.sampling_factories import ss_factory, ess_factory

import data
from fit_parametric_model import fit_parametric_model
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
                                'linear', 'SM'])
    parser.add_argument('--nx', help='(int) number of data points for simulated data',
                        default=400, type=int)
    parser.add_argument('--lengthscale', help='(float) lengthscale for sim kernel',
                        default=2., type=float)
    parser.add_argument('--period', help='(float) period for QP kernel',
                        default=1., type=float)
    parser.add_argument('--slope', help='(float) slope for linear data',
                        default=1., type=float)
    parser.add_argument('--intercept', help='(float) intercept for linear data',
                        default=0., type=float)
    parser.add_argument('--spacing', help='(str) should data be evenly spaced or randomly sampled',
                        default='even', type=str, choices=['even', 'random'])
    parser.add_argument('--noise', help='(bool) should generated data be generated with noise',
                        default='False', type=bool)
    parser.add_argument('--optim_iters', help='(int) number of optimization iterations',
                        default=1, type=int)
    parser.add_argument('--save', help='(bool) save model output (samples and model and omega)?',
                        default=False, type=bool)
    return parser.parse_args()

def main(argv, seed=424):
    '''
    runs ESS with fixed hyperparameters:
    run with -h for CL arguments description
    '''
    # parse CL arguments #
    args = parse()
    gen_pars = [args.lengthscale, args.period]
    linear_pars = [args.slope, args.intercept]

    torch.random.manual_seed(seed)
    ##########################################
    ## some set up and initialization stuff ##
    ##########################################
    ## options ##
    # load data
    train_x, train_y, test_x, test_y, gen_kern = data.read_data(args.data, nx=args.nx, gen_pars=gen_pars,
                                                            linear_pars=linear_pars, spacing='even')
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)
        train_x, train_y, test_x, test_y = train_x.cuda(), train_y.cuda(), test_x.cuda(), test_y.cuda()
        if gen_kern is not None:
            gen_kern = gen_kern.cuda()

    ###########################################
    ## set up the spectral and latent models ##
    ###########################################
    # comput empirical spectral density and omega range
    omega, den = utils.spectral_init(train_x, train_y, args.spacing, num_freq=args.nomg,
            omega_lim=(0., 1.3)
            )
    # omega = omega.squeeze()
    g_init = torch.tensor(den).log()

    # data model #
    data_lh = gpytorch.likelihoods.GaussianLikelihood(noise_prior=gpytorch.priors.SmoothedBoxPrior(1e-8, 1e-3))
    data_mod = spectralgp.models.SpectralModel(train_x, train_y, data_lh, omega=omega, normalize = False,
                            symmetrize = False, transform=utils.link_function)
    data_lh.raw_noise = torch.tensor(-3.5)

    # latent model #
    mu_init = data_mod.covar_module.latent_mean(omega).detach()
    g_init = g_init - mu_init

    latent_lh = gpytorch.likelihoods.GaussianLikelihood(noise_prior=gpytorch.priors.SmoothedBoxPrior(1e-8, 1e-3))
    latent_mod = spectralgp.models.ExactGPModel(omega, g_init, latent_lh)

    # do some pre-training on the demeaned periodogram
    spectralgp.trainer.trainer(omega, g_init, latent_mod, latent_lh)

    ###############################
    ## set up sampling factories ##
    ###############################

    ess_fact = lambda g, h, nsamples : ess_factory(g, h, nsamples, omega, train_x,
                                            train_y, data_mod, data_lh, latent_mod, latent_lh)

    ss_fact = lambda nu, h, nsamples : ss_factory(nu, h, omega, latent_mod, latent_lh,
                                data_mod, data_lh, None, train_x, train_y, nsamples=nsamples)

    ################################
    ## set up alternating sampler ##
    ################################

    alt_sampler = AlternatingSampler(ss_fact, ess_fact,
                totalSamples=args.iters, numInnerSamples=args.ess_iters, numOuterSamples=args.optim_iters,
                h_init = None, g_init = g_init)
    alt_sampler.run()

    data_mod.eval()

    print(list(data_mod.named_parameters()))
    print(list(latent_mod.named_parameters()))

    ###############################
    ## get out parametric models ##
    ###############################

    rbf_mod = fit_parametric_model(train_x, train_y, iters=1000, kernel='rbf')
    matern_mod = fit_parametric_model(train_x, train_y, iters=1000, kernel='matern')
    sm_mod = fit_parametric_model(train_x, train_y, iters=1000, kernel='sm')

    plot_data_para_models(alt_sampler, omega, data_mod, latent_mod,
                                          rbf_mod, matern_mod, sm_mod,
                                          train_x, train_y,
                                          test_x, test_y)

if __name__ == '__main__':
    main(sys.argv[1:])
