import math
import torch
import gpytorch
import seaborn as sns
from scipy.signal import periodogram
from scipy.interpolate import interp1d

import spectralgp

import prcp_utils

import sys

import data_getters

import prcp_plotting
import matplotlib.pyplot as plt
torch.set_default_dtype(torch.float64)
use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)

def main():
    args = prcp_utils.parse()

    ######################################
    ## set up station list and get data ##
    ######################################

    train_stn_list = [703]

    n_stn = len(train_stn_list)
    print('Number of stations: ', n_stn)

    out_list = data_getters.get_data(train_list=train_stn_list, test_list = [],
                                    sample_window=21, standardize=False)
    train_dat = out_list[0]
    sns.set_style("whitegrid")
    sns.set_context("poster")
    # sns.despine()
    plt.plot(train_dat[0, :], marker='.', linestyle="None")
    plt.title("Ithaca Precip")
    plt.xlabel("Days")
    plt.ylabel("Avg. Pos. Precip (0.01 in.)")
    plt.show()
if __name__ == '__main__':
    main()
