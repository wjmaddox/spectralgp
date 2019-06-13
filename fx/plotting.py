import matplotlib.pyplot as plt
import numpy as np
import torch

import math

def plot_train_fits(alt_sampler, train_x_list, train_y_list, train_names, latent_mod, omega,
                    data_mods):
    colors = ["#eac100", "#5893d4", "#10316b", "#070d59"]

    n_samples = min(alt_sampler.gsampled.shape[2], 10)
    #titles = ['Boulder, CO', 'Telluride, CO', 'Steamboat Springs, CO']
    #days_np = days.cpu().numpy()
    #for stn in range(len(train_y_list)):
    for stn in [3,5,8]:
        days_np = train_x_list[stn].cpu().numpy()
        data_mod = data_mods[stn]
        out_samples = alt_sampler.gsampled[stn, :, -n_samples:].detach()
        out_samples = out_samples #+ prcp_utils.get_mean(latent_mod).unsqueeze(1)

        pred_data = torch.zeros(len(train_x_list[stn]), n_samples)
        lower_pred = torch.zeros(len(train_x_list[stn]), n_samples)
        upper_pred = torch.zeros(len(train_x_list[stn]), n_samples)

        data_mod.eval()
        with torch.no_grad():
            for ii in range(n_samples):
                    data_mod.covar_module.latent_params = out_samples[:,ii]
                    data_mod.set_train_data(train_x_list[ii], train_y_list[ii])
                    out = data_mod(train_x_list[stn])
                    lower_pred[:, ii], upper_pred[:, ii] = out.confidence_region()
                    pred_data[:, ii] = out.mean

        pred_data = pred_data.detach().cpu().numpy()
        plt.plot(days_np, pred_data[:, 0],
                color=colors[1], alpha=0.5, label="Mean Prediction")
        plt.plot(days_np, pred_data, color=colors[1], alpha = 0.5)

        for ii in range(n_samples):
            if ii == 0:
                plt.fill_between(days_np, lower_pred[:, ii].cpu().numpy(),
                                upper_pred[:, ii].cpu().numpy(), color=colors[2], alpha = 0.06,
                                label="Confidence Region")
            else:
                plt.fill_between(days_np, lower_pred[:, ii].cpu().numpy(),
                                upper_pred[:, ii].cpu().numpy(), color=colors[2], alpha = 0.06)

        plt.plot(days_np, train_y_list[stn].cpu().numpy(), color=colors[0], label="Data",
                linestyle="None", marker='.', markersize=8, alpha=0.7)
        #plt.title(titles[stn], fontsize=20)
        plt.xlabel('Days',fontsize=14)
        #plt.ylabel('Avg. Pos. Precip (mean-zero)',fontsize=14)
        plt.grid(alpha=0.5)
        plt.legend()
        plt.show()
