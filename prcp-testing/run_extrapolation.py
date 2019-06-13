import math
import torch
import gpytorch

from scipy.signal import periodogram
from scipy.interpolate import interp1d

import spectralgp

import prcp_utils

import sys

import data_getters

import prcp_plotting

torch.set_default_dtype(torch.float64)
use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)

def main(argv, seed=88):
    args = prcp_utils.parse()

    ######################################
    ## set up station list and get data ##
    ######################################

    """VT = [i for i in range(1082, 1090)]
    MA = [i for i in range(401, 413)]
    NH = [i for i in range(631, 636)]
    ME = [i for i in range(374, 385)]
    RI = list(range(921, 923))
    CT = list(range(133, 136))
    NY = list(range(677, 733))
    NJ = list(range(636, 647))
    PA = list(range(897, 920))
    DE = list(range(138,141))
    FL = list(range(142, 164))
    train_stn_list = MA + VT + NH + ME + RI + CT + NY + NJ #+ PA

    # we also tried the entire SE with similar results
    GA = list(range(164, 187))
    AL = list(range(15))
    MS = list(range(470, 502))"""
    stns = [109, 129]
    n_stn = len(stns)

    out_list = data_getters.get_extrapolation_data(train_list=stns, test_cutoff=292,
                                    sample_window=21, standardize=True)
    train_days = torch.tensor(out_list[0]).double()
    train_dat = torch.tensor(out_list[1])
    test_days = torch.tensor(out_list[2]).double()
    test_dat = torch.tensor(out_list[3])
    print(train_dat.size())

    lonlat = torch.tensor(out_list[4])
    names = out_list[5]
    names = [",".join(n.split(",", 2)[:2]) for n in names]
    print("Stations: ", names)

    ####################################################
    ## Define the list of likelihoods and data models ##
    ####################################################

    data_mod_list = []
    data_lh_list = []

    latent_mod = None
    latent_lh = None

    for tt in range(n_stn):
        print(tt)
        data_lh_list.append(gpytorch.likelihoods.GaussianLikelihood(noise_prior=gpytorch.priors.SmoothedBoxPrior(1e-8, 1e-3)))
        data_mod_list.append(spectralgp.models.SpectralModel(train_days, train_dat[tt ,:], likelihood=data_lh_list[tt],
                normalize = False, num_locs = args.nomg, spacing='even', omega_max = args.omega_max,
                latent_mod = latent_mod, latent_lh = latent_lh))
        #data_lh_list[tt].raw_noise = torch.Tensor([-3.5])

        latent_lh, latent_mod = data_mod_list[tt].covar_module.latent_lh, data_mod_list[tt].covar_module.latent_mod
        #latent_mean = data_mod_list[tt].covar_module.latent_mean

    ####################################################################
    ## define the alternating sampler and internal sampling factories ##
    ####################################################################

    alt_sampler = spectralgp.samplers.AlternatingSampler(data_mod_list, data_lh_list,
                        spectralgp.sampling_factories.ss_multmodel_factory,
                        [spectralgp.sampling_factories.ess_factory] * len(stns),
                        numInnerSamples=args.ess_iters, numOuterSamples=args.optim_iters,
                        totalSamples=args.iters,
                        num_dims=1, num_tasks=len(stns)
                        )
    alt_sampler.run()
    omega = latent_mod.train_inputs[0]

    ## save outputs ##
    if args.save:
        fpath = "./saved_outputs/"
        torch.save(omega, fpath + "omega.pt")
        n_samples = min(alt_sampler.gsampled[0].shape[2], 10)
        for stn in range(n_stn):
            samples = alt_sampler.gsampled[0][stn, :, -n_samples:].detach()
            torch.save(samples, fpath + "samples_" + str(stns[stn]) + ".pt")

            torch.save(data_mod_list[stn].state_dict(), fpath + "model_" + str(stns[stn]) + ".pt")

    latent_mod = data_mod_list[0].covar_module.latent_mod
    prcp_plotting.plot_extrapolation(alt_sampler, train_days, train_dat, test_days, test_dat,
                                    latent_mod, omega, data_mod_list, names)
    prcp_plotting.plot_spectrum(alt_sampler, train_days, train_dat, test_days, test_dat,
                                    latent_mod, omega, data_mod_list, names)
if __name__ == '__main__':
    main(sys.argv[1:])
