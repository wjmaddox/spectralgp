import math
import torch
import gpytorch
import argparse


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_stn', help='(int) number of stations to use in training',
                        default = 20, type=int)
    parser.add_argument('--mean', help='(str) latent mean, options = "Constant", "LogRBF"',
                        default="LogRBF")
    parser.add_argument('--nomg', help='(int) number of omegas to use',
                        default=100, type=int)
    parser.add_argument('--iters', help='(int) # of ESS iterations to run',
                        default=100, type=int)
    parser.add_argument('--optim_iters', help='(int) number of optimization iterations',
                        default=1, type=int)
    parser.add_argument('--ess_iters', help='(int) number of ess samples per iteration',
                        default=20, type=int)
    parser.add_argument('--save', help='should you save the models',
                        default=False, type=bool)
    parser.add_argument('--omega_max', help='OMEGA',
                        default=0.2, type=float)
    return parser.parse_args()
