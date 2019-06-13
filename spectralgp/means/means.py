import torch
import gpytorch
from torch.nn.functional import softplus
from gpytorch.priors import NormalPrior

class LogRBFMean(gpytorch.means.Mean):
    """
    Log of an RBF Kernel's spectral density
    """
    def __init__(self, hypers = None):
        super(LogRBFMean, self).__init__()
        if hypers is not None:
            self.register_parameter(name="constant", parameter=torch.nn.Parameter(hypers[-5] + softplus(hypers[-3]).log()))
            self.register_parameter(name="lengthscale", parameter=torch.nn.Parameter(hypers[-4]))
        else:
            self.register_parameter(name="constant", parameter=torch.nn.Parameter(0. * torch.ones(1)))
            self.register_parameter(name="lengthscale", parameter=torch.nn.Parameter(-0.3*torch.ones(1)))

        # register prior
        self.register_prior(name='constant_prior', prior=NormalPrior(torch.zeros(1), 100.*torch.ones(1), transform=None),
            param_or_closure='constant')
        self.register_prior(name='lengthscale_prior', prior=NormalPrior(torch.zeros(1), 100.*torch.ones(1), transform=torch.nn.functional.softplus),
            param_or_closure='lengthscale')

    def set_pars(self, hypers):
        self.constant.data = hypers[-2]
        self.lengthscale.data = hypers[-1]

    def forward(self, input):
        # logrbf up to constants is: c - t^1 / 2l
        out = self.constant - input.pow(2).squeeze(-1) / (2 * (softplus(self.lengthscale.view(-1)) + 1e-7) )
        return out
