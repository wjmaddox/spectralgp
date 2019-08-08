import argparse

from scipy.signal import periodogram, lombscargle
from scipy.interpolate import interp1d
from torch.nn.functional import softplus
import numpy as np
import gpytorch
import torch

import spectralgp

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ess_iters', help='(int) number of ess samples per iteration',
                        default=20, type=int)
    parser.add_argument('--mean', help='(str) latent mean, options = "Constant", "LogRBF"',
                        default="LogRBF")
    parser.add_argument('--nomg', help='(int) number of omegas to use',
                        default=100, type=int)
    parser.add_argument('--iters', help='(int) # of ESS iterations to run',
                        default=100, type=int)
    parser.add_argument('--data', help='(str) options: "airline"...',
                        default='all', type=str,
                        choices=['all', 'challenger', 'fertility', 'concreteslump', 'servo', 'yacht', 'autompg', 'housing', 'stock', 'pendulum', 'energy', 'concrete', 'airfoil'])
    parser.add_argument('--nx', help='(int) number of data points for simulated data',
                        default=400, type=int)
    parser.add_argument('--lengthscale', help='(float) lengthscale for sim kernel',
                        default=2., type=float)
    parser.add_argument('--period', help='(float) period for QP kernel',
                        default=1., type=float)
    parser.add_argument('--slope', help='(float) slope for linear data',
                        default=1., type=float)
    parser.add_argument('--intercept', help='(float) intercept for linear data',
                        default=0., type=float)
    parser.add_argument('--spacing', help='(str) should data be evenly spaced or randomly sampled',
                        default='even', type=str, choices=['even', 'random'])
    parser.add_argument('--noise', help='(bool) should generated data be generated with noise',
                        default='False', type=bool)
    parser.add_argument('--optim_iters', help='(int) number of optimization iterations',
                        default=1, type=int)
    parser.add_argument('--mlatent', help='(str) shared or separate latent gps', default='separate', type=str, choices=['shared', 'separate'])
    parser.add_argument('--model_avg', help='(str) (partial) kernels or (full) kernels + theta model averaging', default='full', type=str, choices=['full', 'partial'])
    parser.add_argument('--omega_max', help='(float) maximum value of omega', default=8., type=float)
    return parser.parse_args()

