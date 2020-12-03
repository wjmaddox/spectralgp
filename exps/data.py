from scipy.io import loadmat

from sklearn.model_selection import train_test_split

import torch
import numpy as np
import pandas as pd
from spectralgp.models import ExactGPModel
import gpytorch


def read_data(dataset, only_scale = False, noise=False, **kwargs):
    kernel=None
    if dataset=='airline' or dataset=='co2' or dataset=='audio1':
        if dataset=='airline':
            loc = "./data/airline.mat"
        elif dataset=='co2':
            loc = './data/CO2.mat'
        else:
            loc = './data/audio1.mat'

        D = loadmat(loc)

        train_y = D['ytrain']
        train_x = D['xtrain']
        test_y = D['ytest']
        test_x = D['xtest']

        train_y = torch.tensor(train_y.astype(float)).view(-1)
        train_x = torch.tensor(train_x.astype(float)).view(-1)
        test_y = torch.tensor(test_y.astype(float)).view(-1)
        test_x = torch.tensor(test_x.astype(float)).view(-1)

        # standardize everything to have 0 mean and variance 1
        x_mean = train_x.mean()
        x_std = train_x.std()

        # standardize x if dataset != audio1
        if dataset != 'audio1':
            train_x = (train_x - x_mean) / x_std

        y_mean = train_y.mean()
        y_std = train_y.std()
        train_y = (train_y - y_mean) / y_std

        if dataset != 'audio1':
            test_x = (test_x - x_mean) / x_std

        test_y = (test_y - y_mean) / y_std

        kernel = None
        y_mean, y_std = y_mean.cpu().numpy(), y_std.cpu().numpy()

    elif dataset == 'jura':
        df = pd.read_csv('./data/jura.txt', delim_whitespace=True)

        msk = np.random.rand(len(df)) < 0.8
        train_df = df[msk]
        test_df = df[~msk]

        train_df = (train_df - train_df.mean())/train_df.std()
        test_df = (test_df - test_df.mean())/test_df.std()

        train_x = torch.tensor(train_df[['X','Y','Rock','Land']].values)#.type(torch.DoubleTensor)
        train_y = torch.tensor(train_df[['Cd','Cu','Pb','Co','Cr','Ni','Zn']].values)#.type(torch.DoubleTensor)

        test_x = torch.tensor(test_df[['X','Y','Rock','Land']].values)#.type(torch.DoubleTensor)
        test_y = torch.tensor(test_df[['Cd','Cu','Pb','Co','Cr','Ni','Zn']].values)#.type(torch.DoubleTensor)

        kernel = None
    elif dataset == 'linear':
        xmin = 0
        xmax = 8
        full_x = torch.linspace(xmin, xmax, kwargs['nx'])
        full_y = kwargs['linear_pars'][0]*full_x + kwargs['linear_pars'][1]

        test_cutoff = int(kwargs['nx']/2)
        train_x = full_x[0:test_cutoff]
        train_y = full_y[0:test_cutoff]
        test_x = full_x[-test_cutoff:]
        test_y = full_y[-test_cutoff:]

    elif dataset == 'sinc':
        # sinc function from wilson & adams, 2013

        sinc = lambda x: torch.sin(np.pi * x) / (np.pi * x)
        xmin=-15.
        xmax=15.
        full_x = torch.linspace(xmin, xmax, kwargs['nx'])

        full_y = sinc(full_x + 10.) + sinc(full_x) + sinc(full_x - 10.)

        test_indices = (full_x > -4.5) * (full_x < 4.5)
        train_x = full_x[~test_indices]
        train_y = full_y[~test_indices]

        test_x = full_x[test_indices]
        test_y = full_y[test_indices]

        y_mean = 0.0
        y_std = 1.0

        kernel = None

    elif dataset in ['QP', 'RBF', 'SM']:
        xmin = -7.
        xmax = 7.
        if dataset=='SM':
            xmin = -10.
            xmax = 10.
        if kwargs['spacing'] == "even":
            full_x = torch.linspace(xmin, xmax, kwargs['nx'])
        elif kwargs['spacing'] == 'random':
            full_x = torch.sort(torch.rand(kwargs['nx']))[0]*xmax

        gen_lh = gpytorch.likelihoods.GaussianLikelihood()
        gen_model = ExactGPModel(train_x=None, train_y=None, likelihood=gen_lh, grid=False)

        ## set the kernel ##
        if dataset=="QP":
            gen_model.covar_module = gpytorch.kernels.ProductKernel(gpytorch.kernels.RBFKernel(),
                                                                    gpytorch.kernels.PeriodicKernel())
            gen_model.covar_module.kernels[0]._set_lengthscale(kwargs['gen_pars'][0])
            gen_model.covar_module.kernels[1]._set_period_length(kwargs['gen_pars'][1])
        elif dataset=='RBF':
            gen_model.covar_module = gpytorch.kernels.RBFKernel()
            gen_model.covar_module._set_lengthscale(kwargs['gen_pars'][0])
        elif dataset=='SM':
            gen_model.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=2)
            gen_model.covar_module._set_mixture_scales(torch.tensor([0.02, 0.01]))
            gen_model.covar_module._set_mixture_weights(torch.tensor([0.4, 0.5]))
            gen_model.covar_module._set_mixture_means(torch.tensor([0.2, 0.9]))

        ## draw training samples ##
        gen_lh.noise = torch.tensor(1.)
        gen_lh.eval()
        gen_model.eval()

        with torch.no_grad():
            if noise:
                print("good")
                print(gen_lh.noise)
                full_y = gen_lh(gen_model(full_x)).rsample().detach()
            else:
                full_y = gen_model(full_x).rsample().detach()

            test_x = full_x
            test_y = full_y

            data_frac = 2.5
            tx1 = full_x[0: int(len(full_x)/data_frac)]
            tx2 = full_x[-int(len(full_x)/data_frac):]
            train_x = torch.cat((tx1, tx2))
            ty1 = full_y[0: int(len(full_y)/data_frac)]
            ty2 = full_y[-int(len(full_y)/data_frac):]
            train_y = torch.cat((ty1, ty2))

        kernel = gen_model.covar_module

        y_mean = train_y.mean()
        y_std = train_y.std()

    elif dataset=='wind':
        from scipy.io import wavfile
        from scipy import signal

        rate, data = wavfile.read('./data/Wind_Original.wav')
        max_time = len(data) / rate

        downsampled_y = torch.DoubleTensor(signal.resample(data, kwargs['nx']))
        print(len(downsampled_y))
        full_x = torch.linspace(0, max_time, kwargs['nx'])

        test_set = (full_x > (0.4 * max_time) ) * (full_x < (0.6 * max_time))
        train_set = -test_set + 1

        test_x = full_x[test_set]
        test_y = downsampled_y[test_set]

        train_x = full_x[train_set]
        train_y = downsampled_y[train_set]

        y_mean = train_y.mean()
        y_std = train_y.std()
        train_y = (train_y - y_mean) / y_std

        #test_x = (test_x - x_mean) / x_std
        test_y = (test_y - y_mean) / y_std

        kernel = None

    elif dataset == 'hr1' or dataset == 'hr2' or dataset=='sunspots':
        if dataset=='hr1':
            test_y = np.loadtxt('data/hr2.txt')
            test_x = (np.linspace(0, 1800,1800))
            dat_size = 350
        elif dataset == 'hr2':
            test_y = np.loadtxt('data/hr1.txt')
            test_x = (np.linspace(0, 1800,1800))
            dat_size = 350
        elif dataset == 'sunspots':
            import statsmodels.api as sm
            dta = sm.datasets.sunspots.load_pandas().data
            test_y = np.array(dta.SUNACTIVITY)[3:]
            y_std = np.std(test_y)
            test_y = test_y/y_std
            test_x = np.array(dta.YEAR)[3:]
            test_x = test_x - np.mean(test_x)
            test_x = test_x/np.std(test_x)
            dat_size = 150

        test_y = test_y - np.mean(test_y)
        indices = np.random.randint(0, len(test_x), size=150)
        indices =np.sort(indices)
        train_y = test_y[indices]
        train_x = test_x[indices]

        train_x = torch.tensor(train_x)
        train_y = torch.tensor(train_y)
        test_x = torch.tensor(test_x)
        test_y = torch.tensor(test_y)

        kernel = None


    # For UCI datasets ( 3droad, gas, yacht, protein, elevators)
    else:
        D = loadmat("./data/{}.mat".format(dataset))
        data = np.array(D['data'])
        train_x, test_x, train_y, test_y = train_test_split(data[:,:-1], data[:,-1], test_size=0.20, random_state=42)

        kernel = None
    if only_scale:
        return y_mean, y_std
    else:
        return train_x, train_y, test_x, test_y, kernel
