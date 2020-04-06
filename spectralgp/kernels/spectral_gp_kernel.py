import math
import torch
import gpytorch
import copy
import spectralgp
from gpytorch.kernels.kernel import Kernel
#from botorch import fit_gpytorch_model
from torch.nn import ModuleList

from ..means import LogRBFMean
from ..utils import spectral_init
from ..models import ExactGPModel
from ..priors import GaussianProcessPrior
from ..trainer import trainer

class SpectralGPKernel(Kernel):
    def __init__(self, omega = None, num_locs = 50, normalize = False, transform = torch.exp, symmetrize = False,
                    **kwargs):
        r"""
        integration: {U, MC} U is trapezoidal rule, MC is Mone Carlo w/ w \sim U(-pi, pi)
        num_locs: number of omegas
        normalize: enforce that integral = 1
        transform: S(\omega) := transform(latent_params(\omega)), default: torch.exp
        symmetrize: use g^*(\omega) := 1/2 * (g(-\omega) + g(\omega))
        """
        super(SpectralGPKernel, self).__init__(**kwargs)

        self.normalize = normalize
        self.transform = transform
        self.symmetrize = symmetrize

        self.num_locs = num_locs


        self.register_parameter('latent_params', torch.nn.Parameter(torch.zeros(self.num_locs)))

    def initialize_from_data(self, train_x, train_y, num_locs=100, spacing='random',
            latent_lh = None, latent_mod = None, period_factor = 8.,
            latent_mean=None, omega=None, **kwargs):
        """
        Spectral methods require a bit of hand-initialization - this is analogous to the SM
        intialize_from_data method.
        train_x: train locations
        train_y: train responses
        num_locs: number of omegas to use
        spacing: type of periodogram to use
        latent_lh: latent model's likliehood function, if non-standard
        latent_mod: latent model
        period_factor: constant to multiply on period
        """
        if omega is None:

            x1 = train_x.unsqueeze(-1)
            x2 = train_x.unsqueeze(-1)
            tau = self.covar_dist(x1, x2, square_dist=False, diag=False,
                                  last_dim_is_batch=False)

            max_tau = torch.max(tau)
            # print(max_tau)
            max_tau = period_factor * max_tau
            omega = math.pi * 2. * torch.arange(self.num_locs).double().div(max_tau)
            # print(omega)

        self.register_parameter('omega', torch.nn.Parameter(omega))
        self.omega.requires_grad = False

        log_periodogram = torch.ones_like(self.omega)

        self.num_locs = len(self.omega)

        # if latent model is passed in, use that
        if latent_lh is None:
            self.latent_lh = gpytorch.likelihoods.GaussianLikelihood(noise_prior=gpytorch.priors.SmoothedBoxPrior(1e-8, 1e-3))
        else:
            print("Using specified latent likelihood")
            self.latent_lh = latent_lh

        if latent_mod is None:
            if latent_mean is None:
                print("Using LogRBF latent mean")
                latent_mean = LogRBFMean
            self.latent_mod = ExactGPModel(self.omega, log_periodogram, self.latent_lh, mean=latent_mean)
            # if isinstance(self.latent_mod.mean_module, spectralgp.means.SM_Mean):
                # self.latent_mod.mean_module.init_from_data(train_x, train_y)
        else:
            print("Using specified latent model")
            self.latent_mod = latent_mod
            #update the training data to include this set of omega and log_periodogram
            self.latent_mod.set_train_data(self.omega, log_periodogram, strict=False)

        self.latent_mod.train()
        self.latent_lh.train()

        # set the latent g to be the demeaned periodogram
        # and make it not require a gradient (we're using ESS for it)
        self.latent_params.data = log_periodogram#self.latent_mod(*self.latent_mod.train_inputs).sample()
        self.latent_params.requires_grad = False

        # clear cache and reset training data
        self.latent_mod.set_train_data(inputs = self.omega, targets=self.latent_params.data, strict=False)

        # register prior for latent_params as latent mod
        latent_prior = GaussianProcessPrior(self.latent_mod, self.latent_lh)
        self.register_prior('latent_prior', latent_prior, lambda: self.latent_params)

        return self.latent_lh, self.latent_mod

    def compute_kernel_values(self, tau, density, normalize=True, fast=False):
        if fast:

            #expand tau \in \mathbb{batch x n x n x M}
            expanded_tau = tau.unsqueeze(-1)
            if len(tau.size()) == 3:
                dims = [1,1,1,self.num_locs]
            else:
                dims = [1,1,self.num_locs]
            expanded_tau = expanded_tau.repeat(*dims)#.float()

            # numerically compute integral in expanded space
            #print(density.device, self.omega.device, expanded_tau.device, expanded_tau.size())
            #print(expanded_tau)
            integrand = density * torch.cos(2.0 * math.pi * self.omega * expanded_tau)

            diff = self.omega[1:] - self.omega[:-1]
            integral = (diff * (integrand[...,1:] + integrand[...,:-1]) / 2.0).sum(-1,keepdim=False)

            # divide by integral of density
            if normalize:
                norm_constant = (diff * (density[1:] + density[:-1]) / 2.0).sum()

                integral = integral / norm_constant


        else:

            integral = torch.zeros_like(tau)
            diff = self.omega[1:] - self.omega[:-1]
            old_integrand = density[0] * torch.cos(2.0 * math.pi * self.omega[0] * tau)

            for i in range(1, len(self.omega)):
                curr_integrand = density[i] * torch.cos(2.0 * math.pi * self.omega[i] * tau)
                integral = integral + diff[i-1] * (curr_integrand + old_integrand) * 0.5
                old_integrand = curr_integrand

            ## NORMALIZATION FROM SSGP ##
            # integral = integral / self.omega.nelement()

            # divide by integral of density
            if normalize:
                norm_constant = (diff * (density[1:] + density[:-1]) / 2.0).sum()
                return integral / norm_constant

        return integral

    def forward(self, x1, x2=None, diag=False, last_dim_is_batch=False,
                **kwargs):
        # print("in fkl x1 = ", x1.shape)
        # print("in fkl x2 = ", x2.shape)
        # print(last_dim_is_batch)
        # print(len(self.omega))
        x1_ = x1
        x2_ = x1 if x2 is None else x2
        if last_dim_is_batch:
            x1_ = x1_.transpose(-1, -2).unsqueeze(-1)
            x2_ = x2_.transpose(-1, -2).unsqueeze(-1)
            tau = x1_ - x2_.transpose(-2, -1)
        else:
            tau = self.covar_dist(x1, x2, square_dist=False, diag=False,
                                  last_dim_is_batch=last_dim_is_batch)

        # print("in FKL x1_", x1_.shape)
        # print("in FKL x2_", x2_.shape)
        # print("in FKL tau = ", tau.shape)

        # transform to enforce positivity
        density = self.transform(self.latent_params)
        #print(self.latent_params, "latent params")
        #print(torch.exp(self.latent_params), "exp(latent params)")
        #print(density, "density")

        if self.symmetrize:
            density = 0.5 * (density + torch.flip(density, [0]))

        output = self.compute_kernel_values(tau, density=density,
                                            normalize=self.normalize, fast=False)
        # print("in FKL output = ", output.shape)
        if diag:
            output = output.diag()
        return output

    def get_latent_mod(self, idx=None):
        return self.latent_mod

    def get_latent_lh(self, idx=None):
        return self.latent_lh

    def get_omega(self, idx=None):
        return self.omega

    def get_latent_params(self, idx=None):
        return self.latent_params

    def set_latent_params(self, g, idx=None):
        #print("here")
        self.latent_params.data = g
