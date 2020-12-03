import math
import torch
import numpy as np


def save_model_output(alt_sampler, data_mod, omega, dat_name,
                      last_samples=10):
    fpath = "./saved_outputs/"
    ## save model ##
    fname = fpath + dat_name + "_model.pt"
    torch.save(data_mod.state_dict(), fname)

    ## save samples ##
    fname = fpath + dat_name + "_samples.pt"
    torch.save(alt_sampler.gsampled[:, -last_samples:].detach(), fname)

    ## save omega ##
    fname = fpath + dat_name + "_omega.pt"
    torch.save(omega, fname)

    return