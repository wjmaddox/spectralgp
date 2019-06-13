import math
import torch
import gpytorch

class SimpleModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(SimpleModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def fit_parametric_model(train_x, train_y, iters=100,
            kernel='rbf'):
    lh = gpytorch.likelihoods.GaussianLikelihood()
    mod = SimpleModel(train_x, train_y, lh)

    if kernel == 'matern':
        mod.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5))
    if kernel == 'QP':
        mod.covar_module = gpytorch.kernels.ProductKernel(gpytorch.kernels.RBFKernel(), gpytorch.kernels.PeriodicKernel())

    if kernel == 'sm':
        ## FOR AIRLINE ##
        mod.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=5)
        mod.covar_module._set_mixture_scales(torch.tensor([0.05, 2., 0.05, 0.1, 0.1]))
        mod.covar_module._set_mixture_weights(torch.tensor([1., 1., 1.,1., 1.]))
        mod.covar_module._set_mixture_means(torch.tensor([1e-4, 1., 2., 8., 5.]))

        ## FOR SINC ##
        # mod.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=9)
        # mod.covar_module._set_mixture_means(torch.tensor([0.01, 0.1, 0.2, 0.4, 1.2]))
        # mod.covar_module._set_mixture_weights(torch.tensor([1., 1., 1., 1., 1.]))
        # mod.covar_module._set_mixture_scales(torch.tensor([0.5, 0.5, 0.5, 0.2, 0.1]))

        # mod.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=2)

    lh.train()
    mod.train()
    optimizer = torch.optim.Adam([
        {'params': mod.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=0.1)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lh, mod)

    for i in range(iters):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from mod
        output = mod(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()

        optimizer.step()

    lh.eval()
    mod.eval()

    return mod, lh
