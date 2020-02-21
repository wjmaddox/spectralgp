import gpytorch
import torch
import math

def get_spectral_density_from_kernel(kernel, locs = 100, base = torch.zeros(1,1), s_range = 4.,
                                    omg=None):
    #this function is another implementation of the trapezoid rule but around K(tau) instead
    # of the spectral density
    if omg is None:
        s = torch.linspace(0, s_range, locs)
    else:
        s = omg
    if isinstance(kernel, gpytorch.kernels.SpectralMixtureKernel):
        dens = torch.zeros_like(s)
        n_mix = kernel.mixture_scales.nelement()
        for ii in range(n_mix):
            norm = torch.distributions.normal.Normal(kernel.mixture_means[ii], kernel.mixture_scales[ii])
            dens = dens + kernel.mixture_weights[ii] * norm.log_prob(s).exp()
            # dens = dens + kernel.mixture_weights[ii]

        return dens.squeeze(), s

    def integrand(tau):
        trig_part = torch.cos(2.0 * math.pi * tau * s)

        kernel_part = kernel(tau, base).evaluate()
        return kernel_part * trig_part


    s_diff = s[1] - s[0]
    tau = torch.linspace(-1 / s_diff, 1 / s_diff, 3 * locs).unsqueeze(1)
    fn_output = integrand(tau)

    # standard trapezoidal rule
    diff = tau[1:] - tau[:-1]
    output = (diff * (fn_output[1:,...] + fn_output[:-1,...])/2.0).sum(0)
    output = torch.clamp(output, 1e-6)
    return 2. * output, s

