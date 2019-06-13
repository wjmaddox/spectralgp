import math
import gpytorch
import torch
import matplotlib.pyplot as plt

def trainer(x, y, latent_mod, latent_lh, training_iter=100):
    latent_lh.train()
    latent_mod.train()

    # print('called')
    optimizer = torch.optim.Adam([
        {'params': latent_mod.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=0.1)

    latent_mod.set_train_data(inputs=x, targets=y, strict = False)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(latent_lh, latent_mod)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = latent_mod(x)
        # Calc loss and backprop gradients
        loss = -mll(output, y)
        loss.backward()

        optimizer.step()

        # if i%10 is 0:
        #     print('Loss is: ', loss.item())


    latent_lh.eval()
    latent_mod.eval()

    ## plotting ##
    # with torch.no_grad():
    #     pred_mean = latent_mod(x).mean
    #     samps = latent_lh(latent_mod(x)).sample(sample_shape=torch.Size((5,)))
    #     plt.plot(x.numpy(), y.numpy(), label="log spectral dens")
    #     plt.plot(x.numpy(), pred_mean.detach().numpy(), label="pred mean")
    #     plt.plot(x.numpy(), samps.t().detach().numpy(), label="samples",
    #                 color='red', alpha=0.2)
    #     plt.title("Latent GP After pre-training")
    #     plt.legend()
    #     plt.show()

    #     plt.plot(x.numpy(), y.exp().numpy(), label="log spectral dens")
    #     plt.plot(x.numpy(), pred_mean.exp().detach().numpy(), label="pred mean")
    #     plt.plot(x.numpy(), samps.exp().t().detach().numpy(), label="samples",
    #                 color='red', alpha=0.2)
    #     plt.title("Spectral Density after pre-training")
    #     plt.legend()
    #     plt.show()
