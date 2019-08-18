import math
import torch
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import sys
#import seaborn as sns
import copy
import numpy as np


def plot_subkernel(alt_sampler, data_mod, dataset, mlatent):
    last_samples = max(10, alt_sampler.gsampled[0].size(2))
    
    color=iter(cm.tab10(np.linspace(0,1,10)))
        
    with torch.no_grad():
        # preprocess the spectral samples #
        data_mod.eval()
        
        plt.figure(figsize=(10,9))
        for dim in range(len(alt_sampler.gsampled)):
            c = next(color)
            tau = torch.stack([torch.linspace(0, 1.5, 600) for dimm in range(len(alt_sampler.gsampled))], dim=1)
            plt_kernels = torch.zeros(tau[:,0].nelement(), last_samples+1)
            for ii in range(last_samples):
                data_mod.covar_module.set_latent_params(alt_sampler.gsampled[dim][0, :, -last_samples:][:,ii], idx=dim)
                plt_kernels[:, ii] = data_mod.covar_module.kernels[dim](tau, torch.zeros(1,int(len(alt_sampler.gsampled)))).evaluate().squeeze(1)
          
            plt_kernels = plt_kernels.detach().cpu().numpy()
            plt_kernels[:, last_samples] = np.mean(plt_kernels[:,0:-1],axis=1)
            
            lower_kern = np.percentile(plt_kernels, 2.5, 1)
            upper_kern = np.percentile(plt_kernels, 97.5, 1)

            tau = tau.cpu().numpy()
            plt.plot(tau[:,0], plt_kernels[:, 0], color=c, alpha=0.35)
            for i in range(last_samples-9, plt_kernels.shape[1]):
                if i == plt_kernels.shape[1]-1:
                    plt.plot(tau[:,0], plt_kernels[:, i], linewidth=3.5, color=c, alpha=1.0, label=r'$d={}$'.format(dim))
                else:
                    plt.plot(tau[:,0], plt_kernels[:, i], color=c, alpha=0.35)
            plt.fill_between(tau[:,0], lower_kern, upper_kern, color=c, alpha=0.1)#, label=r'$\pm 2$ SD')
        plt.ylabel("Covariance", fontsize=22)
        plt.xlabel(r'$\tau$', fontsize=22)
        plt.title("{}".format(dataset), fontsize=26)
        plt.legend(loc=1,prop={'size': 18})
        plt.grid(alpha = 0.5)
        plt.savefig('{}_{}_sampled_posterior_sub_kernels.pdf'.format(dataset, mlatent))
        plt.show()
        plt.close()

        
def plot_prior_subkernel(in_dims, data_mod, dataset, mlatent):
    last_samples = 10
    
    color=iter(cm.tab10(np.linspace(0,1,10)))        
    with torch.no_grad():
        # preprocess the spectral samples #
        data_mod.eval()
        
        plt.figure(figsize=(10,9))
        for dim in range(in_dims):
            c = next(color)
            tau = torch.stack([torch.linspace(0, 1.5, 600) for dimm in range(in_dims)], dim=1)
            plt_kernels = torch.zeros(tau[:,0].nelement(), last_samples+1)
            for ii in range(last_samples):
                train_latent_inputs = data_mod.covar_module.kernels[dim].latent_mod.train_inputs
                sample = data_mod.covar_module.kernels[dim].latent_mod(*train_latent_inputs).sample(sample_shape=torch.Size((1,))).detach()
                data_mod.covar_module.set_latent_params(sample, idx=dim)
                plt_kernels[:, ii] = data_mod.covar_module.kernels[dim](tau, torch.zeros(1,int(in_dims))).evaluate().squeeze(1)

            plt_kernels = plt_kernels.detach().cpu().numpy()
            plt_kernels[:, last_samples] = np.mean(plt_kernels[:,0:-1],axis=1)
            
            lower_kern = np.percentile(plt_kernels, 2.5, 1)
            upper_kern = np.percentile(plt_kernels, 97.5, 1)

            tau = tau.cpu().numpy()
            plt.plot(tau[:,0], plt_kernels[:, 0], color=c, alpha=0.35)
            for i in range(last_samples-9, plt_kernels.shape[1]):
                if i == plt_kernels.shape[1]-1:
                    plt.plot(tau[:,0], plt_kernels[:, i], linewidth=3.5, color=c, alpha=1.0, label=r'$d={}$'.format(dim))
                else:
                    plt.plot(tau[:,0], plt_kernels[:, i], color=c, alpha=0.35)
            plt.fill_between(tau[:,0], lower_kern, upper_kern, color=c, alpha=0.1)#, label=r'$\pm 2$ SD')
        plt.ylabel("Covariance", fontsize=22)
        plt.xlabel(r'$\tau$', fontsize=22)
        plt.title("{}".format(dataset), fontsize=26)
        plt.legend(loc=1,prop={'size': 18})
        plt.grid(alpha = 0.5)
        plt.savefig('{}_{}_sampled_prior_sub_kernels.pdf'.format(dataset, mlatent))
        plt.show()
        plt.close()
        
def plot_subkernel_individual(alt_sampler, data_mod, dataset, mlatent):
    last_samples = max(10, alt_sampler.gsampled[0].size(2))
    
    color=iter(cm.tab10(np.linspace(0,1,10)))
        
    with torch.no_grad():
        # preprocess the spectral samples #
        data_mod.eval()
        
        for dim in range(len(alt_sampler.gsampled)):
            plt.figure(figsize=(10,9))
            c = next(color)
            tau = torch.stack([torch.linspace(0, 1.5, 600) for dimm in range(len(alt_sampler.gsampled))], dim=1)
            plt_kernels = torch.zeros(tau[:,0].nelement(), last_samples+1)
            for ii in range(last_samples):
                data_mod.covar_module.set_latent_params(alt_sampler.gsampled[dim][0, :, -last_samples:][:,ii], idx=dim)
                plt_kernels[:, ii] = data_mod.covar_module.kernels[dim](tau, torch.zeros(1,int(len(alt_sampler.gsampled)))).evaluate().squeeze(1)

            plt_kernels = plt_kernels.detach().cpu().numpy()
            plt_kernels[:, last_samples] = np.mean(plt_kernels[:,0:-1],axis=1)
            
            lower_kern = np.percentile(plt_kernels, 2.5, 1)
            upper_kern = np.percentile(plt_kernels, 97.5, 1)

            tau = tau.cpu().numpy()
            plt.plot(tau[:,0], plt_kernels[:, 0], color=c, alpha=0.35)
            for i in range(last_samples-9, plt_kernels.shape[1]):
                if i == plt_kernels.shape[1]-1:
                    plt.plot(tau[:,0], plt_kernels[:, i], linewidth=3.5, color=c, alpha=1.0, label=r'$d={}$'.format(dim))
                else:
                    plt.plot(tau[:,0], plt_kernels[:, i], color=c, alpha=0.35)
            plt.fill_between(tau[:,0], lower_kern, upper_kern, color=c, alpha=0.1)#, label=r'$\pm 2$ SD')
            plt.ylabel("Covariance", fontsize=22)
            plt.xlabel(r'$\tau$', fontsize=22)
            plt.title("{}".format(dataset), fontsize=26)
            plt.legend(loc=1,prop={'size': 18})
            plt.grid(alpha = 0.5)
            plt.savefig('{}_{}_dimension_{}_sampled_posterior_sub_kernels.pdf'.format(dataset, mlatent, dim))
            plt.show()
            plt.close()

def plot_prior_subkernel_individual(in_dims, data_mod, dataset, mlatent):
    last_samples = 10
    
    color=iter(cm.tab10(np.linspace(0,1,10)))
        
    with torch.no_grad():
        # preprocess the spectral samples #
        data_mod.eval()
        
        for dim in range(in_dims):
            plt.figure(figsize=(10,9))
            c = next(color)
            tau = torch.stack([torch.linspace(0, 1.5, 600) for dimm in range(in_dims)], dim=1)
            plt_kernels = torch.zeros(tau[:,0].nelement(), last_samples+1)
            for ii in range(last_samples):
                train_latent_inputs = data_mod.covar_module.kernels[dim].latent_mod.train_inputs
                sample = data_mod.covar_module.kernels[dim].latent_mod(*train_latent_inputs).sample(sample_shape=torch.Size((1,))).detach()
                data_mod.covar_module.set_latent_params(sample, idx=dim)
                plt_kernels[:, ii] = data_mod.covar_module.kernels[dim](tau, torch.zeros(1,int(in_dims))).evaluate().squeeze(1)
            
            plt_kernels = plt_kernels.detach().cpu().numpy()
            plt_kernels[:, last_samples] = np.mean(plt_kernels[:,0:-1],axis=1)
            
            lower_kern = np.percentile(plt_kernels, 2.5, 1)
            upper_kern = np.percentile(plt_kernels, 97.5, 1)

            tau = tau.cpu().numpy()
            plt.plot(tau[:,0], plt_kernels[:, 0], color=c, alpha=0.35)
            for i in range(last_samples-9, plt_kernels.shape[1]):
                if i == plt_kernels.shape[1]-1:
                    plt.plot(tau[:,0], plt_kernels[:, i], linewidth=3.5, color=c, alpha=1.0, label=r'$d={}$'.format(dim))
                else:
                    plt.plot(tau[:,0], plt_kernels[:, i], color=c, alpha=0.35)
            plt.fill_between(tau[:,0], lower_kern, upper_kern, color=c, alpha=0.1)#, label=r'$\pm 2$ SD')
            plt.ylabel("Covariance", fontsize=22)
            plt.xlabel(r'$\tau$', fontsize=22)
            plt.title("{}".format(dataset), fontsize=26)
            plt.legend(loc=1,prop={'size': 18})
            plt.grid(alpha = 0.5)
            plt.savefig('{}_{}_dimension_{}_sampled_prior_sub_kernels.pdf'.format(dataset, mlatent, dim))
            plt.show()
            plt.close()


def plot_kernel(alt_sampler, data_mod, dataset, mlatent):
    last_samples = max(10, alt_sampler.gsampled[0].size(2))
    print(last_samples)
    with torch.no_grad():
        # preprocess the spectral samples #
        data_mod.eval()
        
        tau = torch.stack([torch.linspace(0, 1.5, 600) for dimm in range(len(alt_sampler.gsampled))], dim=1)
        plt_kernels = torch.zeros(tau[:,0].nelement(), last_samples+1)
        for ii in range(last_samples):
            for dim in range(len(alt_sampler.gsampled)):
                data_mod.covar_module.set_latent_params(alt_sampler.gsampled[dim][0, :, -last_samples:][:,ii], idx=dim)

            plt_kernels[:, ii] = data_mod.covar_module(tau, torch.zeros(1,int(len(alt_sampler.gsampled)))).evaluate().squeeze(1)

        plt_kernels = plt_kernels.detach().cpu().numpy()
        plt_kernels[:, last_samples] = np.mean(plt_kernels[:,0:-1],axis=1)
        lower_kern = np.percentile(plt_kernels, 2.5, 1)
        upper_kern = np.percentile(plt_kernels, 97.5, 1)
        
        tau = tau.cpu().numpy()
        colors = ["#eac100", "#5893d4", "#10316b", "#070d59"]
        plt.figure(figsize=(10,9))
        plt.plot(tau[:,0], plt_kernels[:, 0], color=colors[1], alpha=0.3)
        for i in range(last_samples-9, plt_kernels.shape[1]):
            if i == plt_kernels.shape[1]-1:
                plt.plot(tau[:,0], plt_kernels[:, i], linewidth=3.5, color=colors[1], alpha=1.0)
            else:
                plt.plot(tau[:,0], plt_kernels[:, i], color=colors[1], alpha=0.3)
        plt.fill_between(tau[:,0], lower_kern, upper_kern, color="steelblue", alpha=0.1, label=r'$\pm 2$ SD')
        plt.ylabel("Covariance", fontsize=22)
        plt.xlabel(r'$\tau$', fontsize=22)
        plt.title("{}".format(dataset), fontsize=26)
        plt.legend(loc=1,prop={'size': 18})
        plt.grid(alpha = 0.5)
        plt.savefig('{}_{}_sampled_posterior_kernels.pdf'.format(dataset, mlatent))
        plt.show()
        plt.close()
        #plt.show()
        
def plot_prior_kernel(in_dims, data_mod, dataset, mlatent):
    last_samples = 10
    with torch.no_grad():
        # preprocess the spectral samples #
        data_mod.eval()
        
        tau = torch.stack([torch.linspace(0, 1.5, 600) for dimm in range(in_dims)], dim=1)
        plt_kernels = torch.zeros(tau[:,0].nelement(), last_samples+1)
        for ii in range(last_samples):
            for dim in range(in_dims):
                #prior_draws = prior_latent_mod(omega).sample(sample_shape=torch.Size((n_to_plt,))).detach()
                train_latent_inputs = data_mod.covar_module.kernels[dim].latent_mod.train_inputs
                sample = data_mod.covar_module.kernels[dim].latent_mod(*train_latent_inputs).sample(sample_shape=torch.Size((1,))).detach()
                data_mod.covar_module.set_latent_params(sample, idx=dim)

            plt_kernels[:, ii] = data_mod.covar_module(tau, torch.zeros(1,int(in_dims))).evaluate().squeeze(1)

        plt_kernels = plt_kernels.detach().cpu().numpy()
        plt_kernels[:, last_samples] = np.mean(plt_kernels[:,0:-1],axis=1)
        lower_kern = np.percentile(plt_kernels, 2.5, 1)
        upper_kern = np.percentile(plt_kernels, 97.5, 1)
        
        tau = tau.cpu().numpy()
        colors = ["#eac100", "#5893d4", "#10316b", "#070d59"]
        plt.figure(figsize=(10,9))
        plt.plot(tau[:,0], plt_kernels[:, 0], color=colors[1], alpha=0.3)
        for i in range(last_samples-9, plt_kernels.shape[1]):
            if i == plt_kernels.shape[1]-1:
                plt.plot(tau[:,0], plt_kernels[:, i], linewidth=3.5, color=colors[1], alpha=1.0)
            else:
                plt.plot(tau[:,0], plt_kernels[:, i], color=colors[1], alpha=0.3)
        plt.fill_between(tau[:,0], lower_kern, upper_kern, color="steelblue", alpha=0.1, label=r'$\pm 2$ SD')
        plt.ylabel("Covariance", fontsize=22)
        plt.xlabel(r'$\tau$', fontsize=22)
        plt.title("{}".format(dataset), fontsize=26)
        plt.legend(loc=1,prop={'size': 18})
        plt.grid(alpha = 0.5)
        plt.savefig('{}_{}_sampled_prior_kernels.pdf'.format(dataset, mlatent))
        plt.show()
        plt.close()
