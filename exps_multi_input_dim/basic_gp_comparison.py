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
    def __init__(self, train_x, train_y, likelihood, mean, ard_flag=False):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = mean()
        if ard_flag:
            self.covar_module = mean(ard_num_dims=train_x.size(1))
#        self.covar_module = gpytorch.kernels.RBFKernel(ard_num_dims=train_x.size(1))
#        self.covar_module = gpytorch.kernels.MaternKernel(ard_num_dims=train_x.size(1))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

datasets = ['servo2', 'energy2', 'fertility2', 'yacht2', 'housing2', 'concreteslump2', 'machine2']
rmses =[]
un_rmses =[]
nlls = []
mslls = []

for mm in range(3):
    for datum in datasets:
        for rep in range(10):
            train_x, train_y, test_x, test_y, y_std, y_std_train, gen_kern = data.read_data(datum)

            # initialize likelihood and model
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            # model = ExactGPModel(train_x, train_y, likelihood, gpytorch.kernels.MaternKernel)
            # print(i)
            if mm == 0:
                model = ExactGPModel(train_x, train_y, likelihood, gpytorch.kernels.RBFKernel)
            elif mm == 1:
                model = ExactGPModel(train_x, train_y, likelihood, gpytorch.kernels.RBFKernel, True)
            elif mm == 2:
                model = ExactGPModel(train_x, train_y, likelihood, gpytorch.kernels.MaternKernel, True)

            model.train()
            likelihood.train()

            # Use the adam optimizer
            optimizer = torch.optim.Adam([
                {'params': model.parameters()},  # Includes GaussianLikelihood parameters
            ], lr=0.1)

            # "Loss" for GPs - the marginal log likelihood
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

            training_iter = 200
            for i in range(training_iter):
                # Zero gradients from previous iteration
                optimizer.zero_grad()
                # Output from model
                output = model(train_x)
                # Calc loss and backprop gradients
                loss = -mll(output, train_y)
                loss.backward()

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
                print("Summed NLL: {}".format(nll_sum))
                print("MSLL: {}".format(msll))


            rmses.append(test_rmse)
            un_rmses.append(unnorm_test_rmse)
            nlls.append(float(nll_sum))
            mslls.append(float(msll))

        if mm == 0:
            model_used = 'RBF'
        elif mm == 1:
            model_used = 'ARD'
        elif mm == 2:
            model_used = 'ARD MATERN'
        print(model_used, ' RMSE ', datum, np.around(np.mean(np.array(rmses)), decimals=3), '$\pm$', np.around(np.std(np.array(rmses)), decimals=3))
        print(model_used, ' NLL ', datum, np.around(np.mean(np.array(nlls)), decimals=3), '$\pm$', np.around(np.std(np.array(nlls)), decimals=3))
        print(model_used, ' MSLL ', datum, np.around(np.mean(np.array(mslls)), decimals=3), '$\pm$', np.around(np.std(np.array(mslls)), decimals=3))
        #print(np.around(np.mean(np.array(un_rmses)), decimals=3), '$\pm$', np.around(np.std(np.array(un_rmses)), decimals=3))

        rmses=[]
        un_rmses=[]
        nlls=[]
        mslls=[]
