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
    parser.add_argument('--omega_max', help='(float) maximum value of omega', default=6., type=float)
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
    torch.set_default_tensor_type(torch.DoubleTensor)

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
    ## copied from greg's notebook
    data_lh = gpytorch.likelihoods.GaussianLikelihood(noise_prior=gpytorch.priors.SmoothedBoxPrior(1e-8, 1e-3))
    data_mod = spectralgp.models.SpectralModel(train_x, train_y, data_lh,
        normalize = False, symmetrize = False, num_locs = args.nomg, omega_max = args.omega_max)
    data_lh.raw_noise = torch.tensor(-3.5)


    alt_sampler = spectralgp.samplers.AlternatingSampler([data_mod], [data_lh],
                            spectralgp.sampling_factories.ss_factory,
                            [spectralgp.sampling_factories.ess_factory],
                            totalSamples=args.iters, numInnerSamples=args.ess_iters,
                            numOuterSamples=args.optim_iters)
    alt_sampler.run()

    ###############################
    ## get out parametric models ##
    ###############################

    rbf_mod, _ = fit_parametric_model(train_x, train_y, iters=1000, kernel='rbf')
    matern_mod, _ = fit_parametric_model(train_x, train_y, iters=1000, kernel='matern')
    sm_mod, _ = fit_parametric_model(train_x, train_y, iters=1000, kernel='sm')

    data_mod.eval()
    latent_mod = data_mod.covar_module.latent_mod
    omega = latent_mod.train_inputs[0].squeeze()

    plot_data_para_models(alt_sampler, omega, data_mod, latent_mod,
                                          rbf_mod, matern_mod, sm_mod,
                                          train_x, train_y,
                                          test_x, test_y)

if __name__ == '__main__':
    main(sys.argv[1:])
