import math
import torch
import gpytorch
import numpy as np
from scipy.signal import periodogram
from scipy.interpolate import interp1d

import argparse
import spectralgp


import sys

import data

import plotting

torch.set_default_dtype(torch.float64)

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_stn', help='(int) number of stations to use in training',
                        default = 20, type=int)
    parser.add_argument('--mean', help='(str) latent mean, options = "Constant", "LogRBF"',
                        default="LogRBF")
    parser.add_argument('--dataset', help='(str) dataset', default='fx')
    parser.add_argument('--nomg', help='(int) number of omegas to use',
                        default=100, type=int)
    parser.add_argument('--iters', help='(int) # of ESS iterations to run',
                        default=100, type=int)
    parser.add_argument('--optim_iters', help='(int) number of optimization iterations',
                        default=1, type=int)
    parser.add_argument('--ess_iters', help='(int) number of ess samples per iteration',
                        default=20, type=int)
    parser.add_argument('--spacing', help='(str)',
                        default='random', type=str, choices=['random', 'even'])
    parser.add_argument('--maxomg', help='maximum choice of omega',default=0.2, type=float)
    return parser.parse_args()

def main(argv, seed=88):
    args = parse()

    ######################################
    ## set up station list and get data ##
    ######################################
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)

    train_x, train_y, test_x, test_y = data.get_data(args.dataset, standardize=True)
    n_train = len(train_x)

    mean_module = spectralgp.means.LogRBFMean()

    data_mod_list = []
    lh_list = []
    g_inits = []
    omegas = []
    for tt in range(n_train):
        # construct periodogram
        omega, den = spectralgp.utils.spectral_init(train_x[tt], train_y[tt], args.spacing, num_freq=args.nomg,
                                omega_lim=(1e-6, args.maxomg))
        omegas.append(omega)

        current_logdens = torch.tensor(den).log()

        lh_list.append(gpytorch.likelihoods.GaussianLikelihood())
        current_model = spectralgp.models.SpectralModel(train_x[tt], train_y[tt],
                                          omega=omegas[tt], likelihood=lh_list[tt],
                                          transform=spectralgp.utils.link_function, 
                                          latent_mean=mean_module,
                                          mean=gpytorch.means.ConstantMean(),
                                          normalize=False)
        if use_cuda:
            current_model = current_model.cuda()

        data_mod_list.append(current_model)
        lh_list[tt].raw_noise = torch.tensor(-3.5)

        # demean periodogram
        demeaned_init = current_logdens - data_mod_list[tt].covar_module.latent_mean(omega).detach()
        g_inits.append(demeaned_init)

    g_inits = torch.stack(g_inits)
    print(g_inits.size())

    # set up latent model #
    latent_lh = gpytorch.likelihoods.GaussianLikelihood()
    latent_mod = spectralgp.models.ExactGPModel(omega, g_inits[0], likelihood=latent_lh,
                    mean=gpytorch.means.ZeroMean)
    print(list(data_mod_list[0].named_parameters()))
    #for tt in range(len(omegas)):
    #    spectralgp.trainer.trainer(omegas[tt], g_inits[tt, :], latent_mod, latent_lh)

    ess_list = [(lambda g, h, nsamples : spectralgp.sampling_factories.ess_factory(g, h, nsamples, omega, 
                            train_x[i], train_y[i], data_mod_list[i], lh_list[i], latent_mod, latent_lh))
                            for i in range(n_train)]

    ss_fact = lambda logdens, h, nsamples : spectralgp.sampling_factories.ss_multmodel_factory(logdens, h, omega, latent_mod,
                            latent_lh, data_mod_list, lh_list, train_x, train_y, nsamples)

    alt_sampler = spectralgp.samplers.GibbsAlternatingSampler(ss_fact, ess_list, totalSamples=args.iters,
                                     numInnerSamples=args.ess_iters, numOuterSamples=args.optim_iters,
                                     h_init=None, g_init=g_inits, n_stn=n_train)
    alt_sampler.run()
    print(alt_sampler.gsampled.shape)

    for mod in data_mod_list:
        print(list(mod.named_parameters()))

    # we're only interested in 3,5,8
    smse_full = 0.
    for tt in [3,5,8]:
        data_mod_list[tt].eval()
        data_mod_list[tt].set_train_data(train_x[tt], train_y[tt]) # clear cache

        with gpytorch.settings.fast_pred_var():
            mpred = data_mod_list[tt](test_x[tt]).mean
            smse = torch.mean((mpred.view_as(test_y[tt]) - test_y[tt])**2) / torch.mean((train_y[tt].mean() - test_y[tt])**2)
            print('Model: ', tt, ' SMSE: ', smse)

            smse_full += smse.item()

    ## test some outputs ##
    #plotting.plot_train_fits(alt_sampler, test_x, test_y, None,
    #                                latent_mod, omega, data_mod_list)
    # prcp_plotting.plot_spectrums(alt_sampler, latent_mod, omega)
    print('Total SMSE: ', smse_full/3.)
    return smse_full/3.

if __name__ == '__main__':
    all_smses = []
    for i in range(10):
        smse = main(sys.argv[1:], seed = np.random.randint(100))
        all_smses.append(smse)

    print('Averaeged over 10 seeds: ', np.mean(all_smses), np.std(all_smses))

