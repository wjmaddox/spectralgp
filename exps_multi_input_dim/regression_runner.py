import math
import torch
import gpytorch
import numpy as np
import numpy.linalg as linalg

import spectralgp

from spectralgp.samplers import AlternatingSampler
from spectralgp.models import ExactGPModel, SpectralModel, ProductKernelSpectralModel

from spectralgp.sampling_factories import ss_factory, ess_factory

import data
# from save_models import save_model_output

# import spectralgp.utils as utils
import utils
import argparse

import sys
import matplotlib.pyplot as plt

import traceback

torch.set_default_dtype(torch.float64)

def main(argv, dataset, seed=88):
    '''
    runs ESS with fixed hyperparameters:
    run with -h for CL arguments description
    '''
    # parse CL arguments #
    args = utils.parse()
    gen_pars = [args.lengthscale, args.period]
    linear_pars = [args.slope, args.intercept]
    mlatent = args.mlatent

    # TODO: set seed from main call
    torch.random.manual_seed(seed)
    ##########################################
    ## some set up and initialization stuff ##
    ##########################################

    print("Dataset: {}".format(dataset))
    train_x, train_y, test_x, test_y, y_std, y_std_train, gen_kern = data.read_data(dataset, nx=args.nx, gen_pars=gen_pars,
                                                            linear_pars=linear_pars,
                                                            spacing=args.spacing,
                                                            noise=args.noise)
    in_dims = 1 if train_x.dim() == 1 else train_x.size(1)

    use_cuda = torch.cuda.is_available()
    print('Cuda is available', use_cuda)
    if use_cuda:
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)
        train_x, train_y, test_x, test_y, y_std = train_x.cuda(), train_y.cuda(), test_x.cuda(), test_y.cuda(), y_std.cuda()
        if gen_kern is not None:
            gen_kern = gen_kern.cuda()

    ###########################################
    ## set up the spectral and latent models ##
    ###########################################
    print("Input Dimensions {}".format(in_dims))


    shared = True if mlatent == 'shared' else False

    data_lh = gpytorch.likelihoods.GaussianLikelihood(noise_prior=gpytorch.priors.SmoothedBoxPrior(1e-8, 1e-3))
    data_mod = spectralgp.models.ProductKernelSpectralModel(train_x, train_y, data_lh, shared=shared,
        normalize = False, symmetrize = False, num_locs = args.nomg, spacing=args.spacing, pretrain=False, omega_max = 8., nonstat = True)



    ################################
    ## set up alternating sampler ##
    ################################
    
    alt_sampler = spectralgp.samplers.AlternatingSampler(
    [data_mod], [data_lh], 
    spectralgp.sampling_factories.ss_factory, [spectralgp.sampling_factories.ess_factory],
    totalSamples=args.iters, numInnerSamples=args.ess_iters, numOuterSamples=args.optim_iters, num_dims=in_dims
    )
    alt_sampler.run();

#     alt_sampler = spectralgp.samplers.AlternatingSamplerMultiDim(ss_fact, ess_fact,
#                         totalSamples=args.iters,
#                         numInnerSamples=args.ess_iters,
#                         numOuterSamples=args.optim_iters,
#                         in_dims=in_dims)

#     alt_sampler.run()

    data_mod.eval()
    data_lh.eval()
    
    data_mod_means = torch.zeros_like(data_mod(test_x).mean)
    total_variance = torch.zeros_like(data_lh(data_mod(test_x)).variance)
    
    test_rmse = 0.0
    unnorm_test_rmse = 0.0
    nll_sum = 0.0
    msll = 0.0
    
    with torch.no_grad():
        for x in range(0,alt_sampler.fgsampled[0].shape[-1]):
            for dim in range(0,in_dims):
                data_mod.covar_module.set_latent_params(alt_sampler.fgsampled[dim][0, :, x], idx=dim)
                
            # Verify this is the correct way to handle multidim train setting
            
            data_mod.set_train_data(train_x, train_y) # to clear out the cache
            data_mod_means += data_mod(test_x).mean

            y_preds = data_lh(data_mod(test_x))
            # y_var = f_var + data_noise
            y_var = y_preds.variance
            total_variance += (y_var + torch.pow(data_mod(test_x).mean,2))
    
    
    meaned_data_mod_means = data_mod_means / float(alt_sampler.fgsampled[0].shape[-1])
    total_variance = total_variance/float(alt_sampler.fgsampled[0].shape[-1]) - torch.pow(meaned_data_mod_means,2)
    
    d = meaned_data_mod_means - test_y
    du = d * y_std

    test_rmse = torch.sqrt(torch.mean(torch.pow(d, 2)))
    unnorm_test_rmse = torch.sqrt(torch.mean(torch.pow(du, 2)))
    
    nll = 0.5 * torch.log(2. * math.pi * total_variance) +  torch.pow((meaned_data_mod_means - test_y),2)/(2. * total_variance)
    sll = nll - (0.5 * torch.log(2. * math.pi * torch.pow(y_std_train, 2)) +  torch.pow((torch.mean(train_y) - test_y),2)/(2. * torch.pow(y_std_train, 2)))
    msll += torch.mean(sll)
    nll_sum += nll.sum()
    
    print("Normalised RMSE: {}".format(test_rmse))
    print("Unnormalised RMSE: {}".format(unnorm_test_rmse))
    print("Summed NLL: {}".format(nll_sum))
    print("MSLL: {}".format(msll))

    del data_lh
    del data_mod

    return float(test_rmse), float(unnorm_test_rmse), float(alt_sampler.total_time), float(nll_sum), float(msll)

if __name__ == '__main__':
    args = utils.parse()
    if args.data != 'all':
        # data_l = ['fertility2', 'concreteslump2', 'servo2', 'machine2', 'yacht2', 'housing2', 'energy2']
        # data_l = ['yacht2','housing2','energy2']
        # data_l = ['concreteslump2']
        data_l = [args.data]
        with open('log_file_{}_{}_latent.out'.format(args.mlatent, args.data), 'w+') as f:
            for dataset in data_l:
                try:
                    test_rmses = []
                    unnorm_test_rmses = []
                    times = []
                    nlls = []
                    mslls = []
                    for experiment in range(10):
                        torch.cuda.empty_cache()
                        t, nt, total_times, dnll, dmsll = main(sys.argv[1:], dataset, seed=np.random.randint(10000000))
                        test_rmses.append(t)
                        unnorm_test_rmses.append(nt)
                        times.append(total_times)
                        nlls.append(dnll)
                        mslls.append(dmsll)

                    test_rmses_std = np.around(np.std(np.array(test_rmses)), decimals=3)
                    unnorm_test_rmses_std = np.around(np.std(np.array(unnorm_test_rmses)), decimals=3)

                    test_rmses_mean = np.around(np.mean(np.array(test_rmses)), decimals=3)
                    unnorm_test_rmses_mean = np.around(np.mean(np.array(unnorm_test_rmses)), decimals=3)

                    times_mean = np.around(np.mean(np.array(times)), decimals=3)
                    times_std = np.around(np.std(np.array(times)), decimals=3)

                    nlls_mean = np.around(np.mean(np.array(nlls)), decimals=3)
                    nlls_std = np.around(np.std(np.array(nlls)), decimals=3)

                    mslls_mean = np.around(np.mean(np.array(mslls)), decimals=3)
                    mslls_std = np.around(np.std(np.array(mslls)), decimals=3)

                    f.write("{}; Test RMSE: {} $\pm$ {}\n".format(dataset, test_rmses_mean, test_rmses_std))
                    f.write("{}; Unnormalised Test RMSE: {} $\pm$ {}\n".format(dataset, unnorm_test_rmses_mean, unnorm_test_rmses_std))
                    f.write("{}; NLL: {} $\pm$ {}\n".format(dataset, nlls_mean, nlls_std))
                    f.write("{}; MSLL: {} $\pm$ {}\n".format(dataset, mslls_mean, mslls_std))
                    f.write("{}; Total time: {} $\pm$ {}\n".format(dataset, times_mean, times_std))
                    f.write("Test RMSE: {}\n".format(np.array(test_rmses)))
                    f.write("NLL: {}\n".format(np.array(nlls)))
                    f.write("MSLL: {}\n".format(np.array(mslls)))
                    f.flush()
                except Exception as e:
                    print(e)
                    traceback.print_tb(e.__traceback__)
