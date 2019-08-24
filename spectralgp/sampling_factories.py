import torch
import gpytorch

from .samplers import MeanEllipticalSlice, SGD

# defining ESS factory
def ess_factory(nsamples, data_mod, idx=None):
    # pull out latent model and spectrum from the data model
    omega = data_mod.covar_module.get_omega(idx)
    g_init = data_mod.covar_module.get_latent_params(idx)
    latent_lh = data_mod.covar_module.get_latent_lh(idx)
    latent_mod = data_mod.covar_module.get_latent_mod(idx)

    # update training data
    latent_lh.train()
    latent_mod.train()
    latent_mod.set_train_data(inputs = omega, targets = None, strict = False)

    # draw prior prior distribution
    prior_dist = latent_lh(latent_mod(omega))

    # define a function of the model and log density
    def ess_ell_builder(demeaned_logdens, data_mod):
        with torch.no_grad(), gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=False):
            
            data_mod.covar_module.set_latent_params(demeaned_logdens, idx)
            data_mod.prediction_strategy = None

            loss = data_mod(*data_mod.train_inputs, iters=1, update_labels=False)
            return loss

    # creating model
    return MeanEllipticalSlice(g_init, prior_dist, ess_ell_builder, nsamples, pdf_params=data_mod)


# defining slice sampler factory
def ss_factory(nsamples, data_mod, idx = None):
    if isinstance(data_mod, list):
        data_mod = data_mod[0]
    
    # defining log-likelihood function
    data_mod.train()

    # pull out latent model and spectrum from the data model
    latent_lh = data_mod.covar_module.get_latent_lh(idx)
    latent_mod = data_mod.covar_module.get_latent_mod(idx)
    omega = data_mod.covar_module.get_omega(idx)
    demeaned_logdens = data_mod.covar_module.get_latent_params(idx)

    # update the training inputs
    latent_mod.set_train_data(inputs=omega, targets=demeaned_logdens.detach(), strict=False)

    def ss_ell_builder(latent_mod, latent_lh, data_mod):

        latent_lh.train()
        latent_mod.train()

        with gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=False):
            #loss = data_mll(data_mod(*data_mod.train_inputs), data_mod.train_targets)
            #num_y = len(data_mod.train_targets)
            #print(data_mod(*data_mod.train_inputs))
            #print(data_mod.latent_prior.log_prob(data_mod.latent_params))#/num_y)
            loss = data_mod(*data_mod.train_inputs, iters=1, update_labels=False) + data_mod.covar_module.kernels[idx].latent_prior.log_prob(data_mod.covar_module.kernels[idx].latent_params)#/num_y
            print('Loss is: ', loss)
            return loss, data_mod.state_dict()

    ell_func = lambda h: ss_ell_builder(latent_mod, latent_lh, data_mod)

    pars_for_optimizer = list(data_mod.parameters())
    #for name, param in data_mod.named_parameters():
    #    if param.requires_grad:
    #        print(name, param.data)

    return SGD(pars_for_optimizer, ell_func, n_samples = nsamples, lr=1e-2)