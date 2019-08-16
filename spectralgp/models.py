import math
import torch
import gpytorch
from .means import LogRBFMean
from .priors import GaussianProcessPrior

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


class BayesianLinearRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, num_locs, omega_0 , omega_max, **kwargs):
        super(BayesianLinearRegressionModel, self).__init__(train_x, train_y, gpytorch.likelihoods.GaussianLikelihood())
        
        omega = torch.linspace(omega_0, omega_max, num_locs)
        self.register_parameter('omega', torch.nn.Parameter(omega))
        self.omega.requires_grad = False
        self.omega.data = torch.linspace(omega_0, omega_max, num_locs)
        
        
        log_periodogram = torch.ones_like(self.omega)

        self.num_locs = len(self.omega)
        
        self.register_parameter('latent_params', torch.nn.Parameter(torch.zeros(self.num_locs)))

        self.latent_lh = gpytorch.likelihoods.GaussianLikelihood(noise_prior=gpytorch.priors.SmoothedBoxPrior(1e-8, 1e-3))
        self.latent_mod = ExactGPModel(self.omega, log_periodogram, self.latent_lh, mean=LogRBFMean)

        self.latent_mod.train()
        self.latent_lh.train()

        self.latent_params.data = self.latent_mod(self.omega).sample().detach()#log_periodogram
        self.latent_params.requires_grad = False

        # clear cache and reset training data
        self.latent_mod.set_train_data(inputs = self.omega, targets=self.latent_params.data, strict=False)

        # register prior for latent_params as latent mod
        latent_prior = GaussianProcessPrior(self.latent_mod, self.latent_lh)
        self.register_prior('latent_prior', latent_prior, lambda: self.latent_params)
        
        self.register_parameter('noise', torch.nn.Parameter(torch.randn(1,1)))


    def forward(self, x):
        # need to rearrange latent_params
        #rearranged_latent_params = torch.zeros_like(x[0,:])
        gp_exp_sample = torch.exp(self.latent_params)
        rearranged_latent_params = []
        for omg_id in range(0, gp_exp_sample.size(0)):
            rearranged_latent_params.append(torch.sqrt(gp_exp_sample.data[omg_id]))
            rearranged_latent_params.append(torch.sqrt(gp_exp_sample.data[omg_id]))
        
        rearranged_latent_params = torch.DoubleTensor(rearranged_latent_params)
        
        #print("Train X size:", x.size())
        #print("Rearranged latent params size:", rearranged_latent_params.size())
        #print(torch.matmul(x,rearranged_latent_params))
        out = gpytorch.distributions.MultivariateNormal(torch.matmul(x,rearranged_latent_params), self.noise * torch.eye(x.size(0)))
        return out

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
