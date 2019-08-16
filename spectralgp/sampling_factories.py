import torch
import gpytorch

from .samplers import MeanEllipticalSlice, SGD

# defining ESS factory
def ess_factory(nsamples, data_mod):
    # pull out latent model and spectrum from the data model
    omega = data_mod.omega
    g_init = data_mod.latent_params
    latent_lh = data_mod.latent_lh
    latent_mod = data_mod.latent_mod

    # update training data
    latent_lh.train()
    latent_mod.train()
    latent_mod.set_train_data(inputs = omega, targets = None, strict = False)

    # draw prior prior distribution
    prior_dist = latent_lh(latent_mod(omega))

    # define a function of the model and log density
    def ess_ell_builder(demeaned_logdens, data_mod):
        with torch.no_grad(), gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=False):

            data_mod.latent_params.data = demeaned_logdens
            data_mod.prediction_strategy = None

            loss = data_mod(*data_mod.train_inputs).log_prob(data_mod.train_targets).sum()
            return loss

    # creating model
    return MeanEllipticalSlice(g_init, prior_dist, ess_ell_builder, nsamples, pdf_params=data_mod)

# defining slice sampler factory
def ss_factory(nsamples, data_mod):
    if isinstance(data_mod, list):
        data_mod = data_mod[0]
    
    # defining log-likelihood function
    data_mod.train()

    # pull out latent model and spectrum from the data model
    latent_lh = data_mod.latent_lh
    latent_mod = data_mod.latent_mod
    omega = data_mod.omega
    demeaned_logdens = data_mod.latent_params

    # update the training inputs
    latent_mod.set_train_data(inputs=omega, targets=demeaned_logdens.detach(), strict=False)

    def ss_ell_builder(latent_mod, latent_lh, data_mod):

        latent_lh.train()
        latent_mod.train()

        with gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=False):
            #loss = data_mll(data_mod(*data_mod.train_inputs), data_mod.train_targets)
            num_y = len(data_mod.train_targets)
            #print('P_y is: ', data_mod(*data_mod.train_inputs).log_prob(data_mod.train_targets)/num_y)
            #print('p_nu is: ', data_mod.latent_prior.log_prob(data_mod.latent_params)/num_y)
            loss = data_mod(*data_mod.train_inputs).log_prob(data_mod.train_targets)/num_y + data_mod.latent_prior.log_prob(data_mod.latent_params)/num_y
            print('Loss is: ', loss)
            return loss, data_mod.state_dict()

    ell_func = lambda h: ss_ell_builder(latent_mod, latent_lh, data_mod)

    pars_for_optimizer = list(data_mod.parameters())

    return SGD(pars_for_optimizer, ell_func, n_samples = nsamples, lr=1e-2)