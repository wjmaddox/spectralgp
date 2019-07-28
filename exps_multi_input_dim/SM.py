import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import utils
import data
import numpy as np
import sys

torch.set_default_dtype(torch.float64)


# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, mean):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.base_covar_module = mean(num_mixtures=4, ard_num_dims=train_x.size()[-1])
        self.base_covar_module.initialize_from_data(train_x, train_y)
        self.covar_module = self.base_covar_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

datasets = ['challenger','fertility','concreteslump','servo','yacht','autompg','housing','stock','pendulum','energy','concrete','airfoil']
rmses =[]
un_rmses =[]
nlls = []
mslls = []

reps = 10

for datum in datasets:
    rep = 0
    attempts = 0
    while rep < reps:
        if attempts >= reps:
            break
        try:
            train_x, train_y, test_x, test_y, y_std, y_std_train, gen_kern = data.read_data(datum)
            use_cuda = torch.cuda.is_available()
            #print('Cuda is available', use_cuda)
            if use_cuda:
                torch.set_default_tensor_type(torch.cuda.DoubleTensor)
                train_x, train_y, test_x, test_y, y_std = train_x.cuda(), train_y.cuda(), test_x.cuda(), test_y.cuda(), y_std.cuda()
                if gen_kern is not None:
                    gen_kern = gen_kern.cuda()
            # initialize likelihood and model
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            model = ExactGPModel(train_x, train_y, likelihood, gpytorch.kernels.SpectralMixtureKernel)

            model.train()
            likelihood.train()

            # Use the adam optimizer
            optimizer = torch.optim.Adam([
                {'params': model.parameters()},  # Includes GaussianLikelihood parameters
            ], lr=0.1)

            # "Loss" for GPs - the marginal log likelihood
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

            training_iter = 500
            for i in range(training_iter):
                # Zero gradients from previous iteration
                optimizer.zero_grad()
                # Output from model
                output = model(train_x)
                # Calc loss and backprop gradients
                loss = -mll(output, train_y)
                loss.backward()
                #print(loss)

                optimizer.step()


            # Get into evaluation (predictive posterior) mode
            model.eval()
            likelihood.eval()

            # Test points are regularly spaced along [0,1]
            # Make predictions by feeding model through likelihood
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                d = likelihood(model(test_x)).mean - test_y
                du = d * y_std

                test_rmse = torch.sqrt(torch.mean(torch.pow(d, 2)))
                unnorm_test_rmse = torch.sqrt(torch.mean(torch.pow(du, 2)))

                y_preds = likelihood(model(test_x))
                # y_var = f_var + data_noise
                y_var = y_preds.variance

                nll = 0.5 * torch.log(2. * math.pi * y_var) +  torch.pow((model(test_x).mean - test_y),2)/(2. * y_var)
                sll = nll - (0.5 * torch.log(2. * math.pi * torch.pow(y_std_train, 2)) +  torch.pow((torch.mean(train_y) - test_y),2)/(2. * torch.pow(y_std_train, 2)))
                msll = torch.mean(sll)
                nll_sum = nll.sum()
                #print("Summed NLL: {}".format(nll_sum))
                #print("MSLL: {}".format(msll))

            rmses.append(float(test_rmse))
            un_rmses.append(float(unnorm_test_rmse))
            nlls.append(float(nll_sum))
            mslls.append(float(msll))
            rep = rep + 1

        except:
            attempts += 1
            continue

    model_used = 'Spectral Mixture ARD'

    print(model_used, ' RMSE ', datum, np.around(np.mean(np.array(rmses)), decimals=3), '$\pm$', np.around(np.std(np.array(rmses)), decimals=3))
    print(model_used, ' NLL ', datum, np.around(np.mean(np.array(nlls)), decimals=3), '$\pm$', np.around(np.std(np.array(nlls)), decimals=3))
    print(model_used, ' MSLL ', datum, np.around(np.mean(np.array(mslls)), decimals=3), '$\pm$', np.around(np.std(np.array(mslls)), decimals=3))
    #print(np.around(np.mean(np.array(un_rmses)), decimals=3), '$\pm$', np.around(np.std(np.array(un_rmses)), decimals=3))

    rmses=[]
    un_rmses=[]
    nlls=[]
    mslls=[]
