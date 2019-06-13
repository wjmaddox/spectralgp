import gpytorch
import torch

from scipy.signal import periodogram, welch
from scipy.interpolate import interp1d

def get_kernels(latent_draws, tau, spec_model, train_x, train_y):
    n_draws = latent_draws.shape[0]
    kernels = torch.zeros(n_draws, tau.nelement())

    for draw in range(n_draws):
        spec_model.covar_module.set_latent_params(latent_draws[draw, :])
        spec_model.set_train_data(train_x, train_y)
        kernels[draw, :] = spec_model.covar_module(tau, torch.zeros(1,1)).evaluate().squeeze()

    return kernels

def update_parameters(object, params_to_update, param_dict):
    for param in params_to_update.keys():
        param_dict[param] = params_to_update[param]
    return param_dict

def spectral_init(x, y, spacing, num_freq=500, omega_lim=None,
                    rtn_oneside=True):
    if spacing == 'even':
        f, den = periodogram(y.cpu().numpy(), fs=1/(x[1] - x[0]).item(),
                            return_onesided=rtn_oneside)
    elif spacing == 'random':
        interp_x = torch.linspace(x.min(), x.max()-1e-6, num_freq)
        interp_y = interp1d(x.cpu().numpy(), y.cpu().numpy())(interp_x.cpu().numpy())

        cand = 0.0
        idx = 0
        while cand < 1.e-23:
            cand = (interp_x[idx+1] - interp_x[idx])
            idx += 1
        f, den = periodogram(interp_y, fs=1/(cand).item(), return_onesided=rtn_oneside)

    ## get omegas ##
    print(omega_lim)
    if omega_lim is not None:
        lw_bd = max(min(f), omega_lim[0])
        up_bd = min(max(f), omega_lim[1])
        omega = torch.linspace(lw_bd + 1e-7, up_bd - 1e-7, num_freq)
    else:
        omega = torch.linspace(min(f)+1e-6, 0.5 * (max(f)-1e-6), num_freq)

    interp_den = interp1d(f, den)(omega.cpu().numpy())
    interp_den += 1e-10
    log_periodogram = torch.Tensor(interp_den).log()

    return omega, log_periodogram
