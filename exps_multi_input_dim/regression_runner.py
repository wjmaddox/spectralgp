import math
import torch
import gpytorch
import numpy as np
import numpy.linalg as linalg

import spectralgp

from spectralgp.samplers import AlternatingSampler
from spectralgp.models import ExactGPModel, SpectralModel, ProductKernelSpectralModel

from spectralgp.sampling_factories import ss_factory, ess_factory
from custom_plotting import plot_kernel

import data
# from save_models import save_model_output

# import spectralgp.utils as utils
import utils
import argparse

import sys
import matplotlib.pyplot as plt

import traceback

torch.set_default_dtype(torch.float64)

def model_average(data_mod, alt_sampler, train_x, train_y, test_x, in_dims, state="full"):
    data_mod.eval()
    data_mod_means = torch.zeros_like(data_mod(test_x).mean)
    total_variance = torch.zeros_like(data_mod(test_x).variance)
    with torch.no_grad():
        marg_samples_num = min(len(alt_sampler.fhsampled[0][0]), alt_sampler.fgsampled[0].shape[-1])
        for x in range(0, marg_samples_num):
            if state == "full":
                print("densities + thetas model averaging")
                # This line must come first
                data_mod.load_state_dict(alt_sampler.fhsampled[0][0][x]) # dim, ???, nsample
            else:
                print("densities model averaging")
            for dim in range(0,in_dims):
                data_mod.latent_params.data = alt_sampler.fgsampled[dim][0, :, x]
            data_mod.set_train_data(train_x, train_y) # to clear out the cache
            data_mod_means += data_mod(test_x).mean
            y_preds = data_mod(test_x)
            # y_var = f_var + data_noise
            y_var = y_preds.variance
            total_variance += (y_var + torch.pow(data_mod(test_x).mean,2))
    meaned_data_mod_means = data_mod_means / float(marg_samples_num)
    total_variance = total_variance/float(marg_samples_num) - torch.pow(meaned_data_mod_means,2)
    
    return meaned_data_mod_means, total_variance

def main(argv, dataset, seed, iteration):
    '''
    runs ESS with fixed hyperparameters:
    run with -h for CL arguments description
    '''
    # parse CL arguments #
    args = utils.parse()
    gen_pars = [args.lengthscale, args.period]
    linear_pars = [args.slope, args.intercept]
    mlatent = args.mlatent
    model_avg = args.model_avg

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

    ###########################################
    ## set up the spectral and latent models ##
    ###########################################
    print("Input Dimensions {}".format(in_dims))


    shared = True if mlatent == 'shared' else False
    
    data_x = torch.linspace(0, 5, 500)
    # True function is sin(2*pi*x) with Gaussian noise
    data_y = (4. + torch.sin(data_x * (2 * math.pi))) + torch.randn(data_x.size()) * 0.2
    
    omega_max = 8.
    omega_data = torch.linspace(1.e-10, omega_max, args.nomg)
    omega_delta = omega_data[1] - omega_data[0]
    phi_const = torch.sqrt(omega_delta/2.)
    
    mapped_data_x = []
    for obs_id in range(data_x.size(0)):
        mapped_data_x.append([])
        for omg_id in range(0, args.nomg):
            if omg_id == 0 or omg_id == args.nomg-1:
                mapped_data_x[obs_id].append(torch.cos(2.0*math.pi*omega_data[omg_id]*data_x[obs_id]))
                mapped_data_x[obs_id].append(torch.sin(2.0*math.pi*omega_data[omg_id]*data_x[obs_id]))
            else:
                mapped_data_x[obs_id].append(math.sqrt(2.0)*torch.cos(2.0*math.pi*omega_data[omg_id]*data_x[obs_id]))
                mapped_data_x[obs_id].append(math.sqrt(2.0)*torch.sin(2.0*math.pi*omega_data[omg_id]*data_x[obs_id]))
    
    data_x = torch.DoubleTensor(mapped_data_x) * phi_const
    
    train_x = data_x[0:250,:]
    train_y = data_y[0:250]
    
    test_x = data_x[250:,:]
    test_y = data_y[250:]
    
    print(train_x.size(), train_y.size())
    print(test_x.size(), test_y.size())
    
    
#     mapped_train_x = []
#     mapped_test_x = []
#     for obs_id in range(train_x.size(0)):
#         mapped_train_x.append([])
#         for omg_id in range(0, args.nomg):
#             if omg_id == 0 or omg_id == args.nomg-1:
#                 mapped_train_x[obs_id].append(torch.cos(2.0*math.pi*omega_data[omg_id]*train_x[obs_id]))
#                 mapped_train_x[obs_id].append(torch.sin(2.0*math.pi*omega_data[omg_id]*train_x[obs_id]))
#             else:
#                 mapped_train_x[obs_id].append(math.sqrt(2.0)*torch.cos(2.0*math.pi*omega_data[omg_id]*train_x[obs_id]))
#                 mapped_train_x[obs_id].append(math.sqrt(2.0)*torch.sin(2.0*math.pi*omega_data[omg_id]*train_x[obs_id]))
    
#     for obs_id in range(test_x.size(0)):
#         mapped_test_x.append([])
#         for omg_id in range(0, args.nomg):
#             if omg_id == 0 or omg_id == args.nomg-1:
#                 mapped_test_x[obs_id].append(torch.cos(2.0*math.pi*omega_data[omg_id]*test_x[obs_id]))
#                 mapped_test_x[obs_id].append(torch.sin(2.0*math.pi*omega_data[omg_id]*test_x[obs_id]))
#             else:
#                 mapped_test_x[obs_id].append(math.sqrt(2.0)*torch.cos(2.0*math.pi*omega_data[omg_id]*test_x[obs_id]))
#                 mapped_test_x[obs_id].append(math.sqrt(2.0)*torch.sin(2.0*math.pi*omega_data[omg_id]*test_x[obs_id]))
    
#     train_x = torch.DoubleTensor(mapped_train_x)
#     test_x = torch.DoubleTensor(mapped_test_x)
    
#     train_x = phi_const * train_x
#     test_x = phi_const * test_x
    
    print(train_x.size(), train_y.size())
    print(test_x.size(), test_y.size())
    
    data_mod = spectralgp.models.BayesianLinearRegressionModel(train_x, train_y, args.nomg, omega_max)



    ################################
    ## set up alternating sampler ##
    ################################

    alt_sampler = spectralgp.samplers.AlternatingSampler(
    [data_mod],
    spectralgp.sampling_factories.ss_factory, [spectralgp.sampling_factories.ess_factory],
    totalSamples=args.iters, numInnerSamples=args.ess_iters, numOuterSamples=args.optim_iters, num_dims=in_dims
    )
    alt_sampler.run()
    
    #meaned_data_mod_means, total_variance = model_average(data_mod, alt_sampler, train_x, train_y, test_x, in_dims, model_avg)

    data_mod.eval()

    d = data_mod(test_x).mean - test_y
    du = d * y_std
    
    test_rmse = torch.sqrt(torch.mean(torch.pow(d, 2)))
    unnorm_test_rmse = torch.sqrt(torch.mean(torch.pow(du, 2)))

    print("Normalised RMSE: {}".format(test_rmse))
    print("Unnormalised RMSE: {}".format(unnorm_test_rmse))
    
    print(data_x.size(), data_y.size())
    print(train_x.size(), test_x.size(), data_x.size())
    print(data_mod(data_x).mean)

    plt.plot(data_y.numpy(), label='data')
    plt.plot(train_y.numpy(), marker='o', label='train')
    #plt.plot(test_y.numpy(), marker='.', label='test')
    plt.plot(data_mod(data_x).mean.detach().numpy(), marker='*', label='BLR')
    plt.legend()
    plt.savefig("BLR_out.png")
    
    print(data_mod(data_x).mean.detach().numpy())
    #plot_kernel(alt_sampler, data_mod, dataset, mlatent)

    del data_mod

    #return float(test_rmse), float(unnorm_test_rmse), float(alt_sampler.total_time), float(nll_sum), float(msll)
    return float(test_rmse), float(unnorm_test_rmse), float(alt_sampler.total_time), -1.0, -1.0

if __name__ == '__main__':
    args = utils.parse()
    if args.data != 'all':
        data_l = [args.data]
        with open('log_file_{}_{}_modelavg_{}_latent.out'.format(args.mlatent, args.data, args.model_avg), 'w+') as f:
            for dataset in data_l:
                try:
                    test_rmses = []
                    unnorm_test_rmses = []
                    times = []
                    nlls = []
                    mslls = []
                    for experiment in range(1):
                        torch.cuda.empty_cache()
                        t, nt, total_times, dnll, dmsll = main(sys.argv[1:], dataset, seed=np.random.randint(10000000), iteration=experiment)
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
