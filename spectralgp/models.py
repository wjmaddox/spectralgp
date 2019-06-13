import math
import torch
import gpytorch

# We will use the simplest form of GP model, exact inference
# TODO: set transforms
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, mean=gpytorch.means.ZeroMean, grid = True):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = mean()

        self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=1.5,
                    lengthscale_prior = gpytorch.priors.NormalPrior(torch.zeros(1), torch.ones(1),
                                            transform=torch.exp)
                    ),
                    outputscale_prior = gpytorch.priors.NormalPrior(torch.zeros(1), torch.ones(1),
                                            transform=torch.exp)
                )

        self.grid = grid
        if self.grid:
            self.grid_module = gpytorch.kernels.GridKernel(self.covar_module, grid = train_x.unsqueeze(1))

    def forward(self, x):
        mean_x = self.mean_module(x)
        if self.grid:
            covar_x = self.grid_module(x)
        else:
            covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class LatentGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, mean=gpytorch.means.ZeroMean, grid = True):
        super(LatentGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = mean()

        self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=1.5,
                    lengthscale_prior = gpytorch.priors.NormalPrior(torch.zeros(1), torch.ones(1),
                                            transform=torch.exp)
                    ),
                    outputscale_prior = gpytorch.priors.NormalPrior(torch.zeros(1), torch.ones(1),
                                            transform=torch.exp)
                )

        self.grid = grid
        if self.grid and train_x is not None:
            self.grid_module = gpytorch.kernels.GridKernel(self.covar_module,
                                                           grid = train_x.unsqueeze(1))
        else:
            self.grid_module = None
    def forward(self, x):
        mean_x = self.mean_module(x)
        if self.grid and self.train_inputs is not None:
            if self.grid_module is None:
                self.grid_module = gpytorch.kernels.GridKernel(self.covar_module,
                                                               grid = self.train_inputs[0].unsqueeze(1))
            covar_x = self.grid_module(x)
        else:
            covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class SpectralModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, **kwargs):
        from spectralgp.kernels import SpectralGPKernel

        super(SpectralModel, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()

        self.covar_module = SpectralGPKernel(**kwargs)

        self.covar_module.initialize_from_data(train_x, train_y, **kwargs)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ProductKernelSpectralModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, shared, **kwargs):
        from .kernels import ProductSpectralGPKernel

        super(ProductKernelSpectralModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = ProductSpectralGPKernel(train_x, train_y, shared, **kwargs)


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



class GridSpectralModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, omega=None, mean=gpytorch.means.ConstantMean, normalize = False, **kwargs):
        super(GridSpectralModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = mean()
        self.covar_module = SpectralGPKernel(omega=omega, normalize=normalize, **kwargs)
        self.normalize = normalize
        if self.normalize:
            self.scale_module = gpytorch.kernels.ScaleKernel(self.covar_module,
                                                outputscale_prior = gpytorch.priors.NormalPrior(torch.zeros(1), 100.*torch.ones(1),
                                                                            transform=torch.exp)
                                                                            )

            kern = self.scale_module
        else:
            kern = self.covar_module

        #grid_size = gpytorch.utils.grid.choose_grid_size(train_x, ratio=0.1)
        self.grid_module = gpytorch.kernels.GridInterpolationKernel(kern, grid_size=min(100, len(train_x)), num_dims=1)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.grid_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
