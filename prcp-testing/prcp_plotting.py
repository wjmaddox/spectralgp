import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch

import math
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['legend.fontsize'] = 22

def plot_train_fits(alt_sampler, days, train_dat, train_names, latent_mod, omega,
                    data_mods):

    colors = cm.get_cmap("tab10")
    n_samples = min(alt_sampler.gsampled[0].shape[2], 10)
    titles = ['Boulder, CO', 'Telluride, CO', 'Steamboat Springs, CO']
    days_np = days.cpu().numpy()
    for stn in range(train_dat.shape[0]):
        data_mod = data_mods[stn]
        out_samples = alt_sampler.gsampled[0][stn, :, -n_samples:].detach()
        out_samples = out_samples #+ prcp_utils.get_mean(latent_mod).unsqueeze(1)

        pred_data = torch.zeros(len(days), n_samples)
        lower_pred = torch.zeros(len(days), n_samples)
        upper_pred = torch.zeros(len(days), n_samples)

        with torch.no_grad():
            for ii in range(n_samples):
                    data_mod.covar_module.set_latent_params(out_samples[:,ii])
                    data_mod.set_train_data(days, train_dat[stn, :])
                    data_mod.eval()
                    out = data_mod(days)
                    lower_pred[:, ii], upper_pred[:, ii] = out.confidence_region()
                    pred_data[:, ii] = out.mean

        pred_data = pred_data.detach().cpu().numpy()
        plt.figure(figsize=(10,9))
        plt.plot(days_np, pred_data[:, 0],
                color=colors(0), alpha=0.5, label="Mean Prediction")
        plt.plot(days_np, pred_data, color=colors(0), alpha = 0.5)

        for ii in range(n_samples):
            if ii == 0:
                plt.fill_between(days_np, lower_pred[:, ii].cpu().numpy(),
                                upper_pred[:, ii].cpu().numpy(), color=colors(0), alpha = 0.06,
                                label="CR")
            else:
                plt.fill_between(days_np, lower_pred[:, ii].cpu().numpy(),
                                upper_pred[:, ii].cpu().numpy(), color=colors(0), alpha = 0.06)

        plt.plot(days_np, train_dat[stn, :].numpy(), color=colors(1), label="Data",
                linestyle="None", marker='.', markersize=16, alpha=0.7)
        plt.title(titles[stn], fontsize=24)
        plt.xlabel('Days',fontsize=20)
        plt.ylabel('Avg. Pos. Precip (mean-zero)',fontsize=20)
        plt.grid(alpha=0.5)

        if stn == 0:
            plt.legend()
        BIGGER_SIZE = 18
        # plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
        # plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
        # plt.rc('legend', fontsize=20)
        plt.show()

def plot_extrapolation(alt_sampler, train_days, train_dat, test_days, test_dat,
                                 latent_mod, omega, data_mod_list, names):


    colors = cm.get_cmap("tab10")
    n_samples = min(alt_sampler.gsampled[0].shape[2], 10)
    full_days = torch.cat((train_days, test_days)).sort()[0]
    full_days_np = full_days.cpu().numpy()
    for stn in range(train_dat.shape[0]):
        data_mod = data_mod_list[stn]
        out_samples = alt_sampler.gsampled[0][stn, :, -n_samples:].detach()
        out_samples = out_samples #+ prcp_utils.get_mean(latent_mod).unsqueeze(1)

        pred_data = torch.zeros(len(full_days), n_samples)
        lower_pred = torch.zeros(len(full_days), n_samples)
        upper_pred = torch.zeros(len(full_days), n_samples)

        data_mod.eval()
        with torch.no_grad():
            for ii in range(n_samples):
                    data_mod.covar_module.set_latent_params(out_samples[:,ii])
                    data_mod.set_train_data(train_days, train_dat[stn, :])
                    data_mod.eval()
                    out = data_mod(full_days)
                    lower_pred[:, ii], upper_pred[:, ii] = out.confidence_region()
                    pred_data[:, ii] = out.mean

        pred_data = pred_data.detach().cpu().numpy()
        plt.figure(figsize=(10,9))
        plt.plot(full_days_np, pred_data[:, 0],
                color=colors(0), alpha=0.5, label="Mean Prediction")
        plt.plot(full_days_np, pred_data, color=colors(0), alpha = 0.5)

        for ii in range(n_samples):
            if ii == 0:
                plt.fill_between(full_days_np, lower_pred[:, ii].cpu().numpy(),
                                upper_pred[:, ii].cpu().numpy(), color=colors(0), alpha = 0.06,
                                label="Confidence Region")
            else:
                plt.fill_between(full_days_np, lower_pred[:, ii].cpu().numpy(),
                                upper_pred[:, ii].cpu().numpy(), color=colors(0), alpha = 0.06)

        plt.plot(train_days.cpu().numpy(), train_dat[stn, :].cpu().numpy(), color=colors(1), label="Data",
                alpha=0.7, linewidth=3.5)
        plt.plot(test_days.cpu().numpy(), test_dat[stn, :].cpu().numpy(), color=colors(1), label="Test",
                linestyle="None", marker='.', markersize=16, alpha=0.7)
        plt.title(names[stn], fontsize=28)
        plt.xlabel('Days',fontsize=24)
        plt.ylabel('Avg. Pos. Precip (mean-zero)',fontsize=28)
        plt.grid(alpha=0.5)
        if stn==0:
            plt.legend(loc=2)
        plt.show()

def plot_spectrum(alt_sampler, train_days, train_dat, test_days, test_dat,
                                 latent_mod, omega, data_mod_list, names):


    colors = cm.get_cmap("tab10")
    n_samples = min(alt_sampler.gsampled[0].shape[2], 10)

    plt.figure(figsize=(10,9))
    for stn in range(train_dat.shape[0]):
        data_mod = data_mod_list[stn]

        with torch.no_grad():
            out_samples = alt_sampler.gsampled[0][stn, :, -n_samples:].detach()

            if stn == 0:
                omega = data_mod.covar_module.omega

                pred_mean = data_mod.covar_module.latent_mod(omega).mean.unsqueeze(1)
                plt.plot(omega.cpu().numpy(), pred_mean.exp().cpu().numpy(),
                        color=colors(2*stn), alpha=0.5, label="Mean ")

            log_samples = out_samples + pred_mean
            samples = torch.exp(log_samples).cpu().numpy()
            #omega = omega.cpu().numpy()

        for ii in range(n_samples):
            if ii == 0:
                plt.plot(omega.cpu().numpy(), samples[:,ii], color=colors(2*stn+1), alpha = 0.15, label = "Samples "+names[stn])
            else:
                plt.plot(omega.cpu().numpy(), samples[:,ii], color=colors(2*stn+1), alpha = 0.15)

    plt.title('All Spectral Densities', fontsize=28)
    plt.xlabel('Omega',fontsize=24)
    plt.ylabel('S(Omega)',fontsize=28)
    plt.grid(alpha=0.5)
    plt.legend()
    plt.show()
