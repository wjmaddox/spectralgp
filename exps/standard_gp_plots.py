import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import seaborn as sns
import data
import sys
import argparse
torch.set_default_dtype(torch.float64)

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ess_iters', help='(int) number of ess samples per iteration',
                        default=20, type=int)
    parser.add_argument('--mean', help='(str) latent mean, options = "Constant", "LogRBF"',
                        default="LogRBF")
    parser.add_argument('--nomg', help='(int) number of omegas to use',
                        default=100, type=int)
    parser.add_argument('--iters', help='(int) # of ESS iterations to run',
                        default=100, type=int)
    parser.add_argument('--data', help='(str) options: "airline"...',
                        default='airline', type=str,
                        choices=['airline', 'jura', 'QP', 'audio1', 'RBF', 'co2', 'sinc', 'wind',
                                'gas', 'yacht', '3droad', 'electric', 'protein', 'video', 'elevators',
                                'linear', 'SM', 'hr1', 'hr2', 'sunspots'])
    parser.add_argument('--nx', help='(int) number of data points for simulated data',
                        default=400, type=int)
    parser.add_argument('--lengthscale', help='(float) lengthscale for sim kernel',
                        default=2., type=float)
    parser.add_argument('--period', help='(float) period for QP kernel',
                        default=1., type=float)
    parser.add_argument('--noise', help='(bool) should generated data be generated with noise',
                        default='False', type=bool)
    parser.add_argument('--optim_iters', help='(int) number of optimization iterations',
                        default=1, type=int)
    parser.add_argument('--omega_max', help='(float) maximum value of omega', default=1., type=float)
    parser.add_argument('--save', help='(bool) save model output (samples and model and omega)?',
                        default=False, type=bool)
    parser.add_argument('--spacing', help='(str) should data be evenly spaced or randomly sampled',
                        default='even', type=str, choices=['even', 'random'])
    return parser.parse_args()

def main(argv, seed=88):
    args = parse()
    n_plt = 5
    gen_pars = [args.lengthscale, args.period]
    torch.random.manual_seed(seed)

    ## generate data ##
    gen_lh = gpytorch.likelihoods.GaussianLikelihood()
    gen_mod = ExactGPModel(None, None, gen_lh)
    gen_mod.covar_module.base_kernel._set_lengthscale(gen_pars[0])
    # gen_mod.covar_module = gpytorch.kernels.ProductKernel(gpytorch.kernels.RBFKernel(), gpytorch.kernels.PeriodicKernel())
    # gen_mod.covar_module.kernels[0]._set_lengthscale(gen_pars[0])
    # gen_mod.covar_module.kernels[1]._set_period_length(gen_pars[1])
    gen_lh.eval()
    gen_mod.eval()

    train_x = torch.linspace(0, 5, args.nx)
    train_y = gen_mod(train_x).sample(sample_shape=torch.Size((1,))).squeeze().detach()

    ## set up actual model ##
    lh = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, lh)

    lh.train()
    model.train()

    samples = model(train_x).sample(sample_shape=torch.Size((n_plt, ))).t().detach()
    lower, upper = lh(model(train_x)).confidence_region()

    sns.set_style("whitegrid")
    sns.set_context('talk')
    sns.despine()
    sns.set_palette("muted")
    plt.figure(figsize=(10,9))
    plt.plot(train_x.numpy(), samples.numpy())
    plt.plot(train_x.numpy(), train_y.numpy(), linewidth=2.5, linestyle="None",
             marker=".", label="Data")
    # plt.fill_between(train_x.numpy(), lower.detach().numpy(), upper.detach().numpy(),
    #                  color="steelblue", alpha=0.1, label=r'$\pm 2$ SD')
    plt.xlim((0, 5))
    plt.ylim((-2,2))
    plt.xlabel("X", fontsize=28)
    plt.ylabel("Y", fontsize=28)
    plt.title("Prior Draws", fontsize=28)
    plt.legend()
    plt.show()


    ## now do training ##
    # Find optimal model hyperparameters
    model.train()
    lh.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=0.1)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lh, model)

    training_iter = 40
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        # print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        #     i + 1, training_iter, loss.item(),
        #     model.covar_module.base_kernel.lengthscale.item(),
        #     model.likelihood.noise.item()
        # ))
        optimizer.step()

    lh.eval()
    model.eval()

    preds = model(train_x).sample(sample_shape=torch.Size((n_plt, ))).t().detach()
    lower, upper = model(train_x).confidence_region()

    plt.figure(figsize=(10, 9))
    plt.plot(train_x.numpy(), preds.numpy())
    plt.plot(train_x.numpy(), train_y.numpy(), linewidth=2.5, linestyle="None",
             marker=".", label="Data")
    plt.fill_between(train_x.numpy(), lower.detach().numpy(), upper.detach().numpy(),
                     color="steelblue", alpha=0.1, label=r'$\pm 2$ SD')
    plt.xlim((0, 5))
    plt.ylim((-0.75, 0.5))
    plt.xlabel("X", fontsize=28)
    plt.ylabel("Y", fontsize=28)
    plt.title("Posterior Draws", fontsize=28)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main(sys.argv[1:])
