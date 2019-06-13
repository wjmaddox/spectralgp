from scipy.io import loadmat

from sklearn.model_selection import train_test_split

import torch
import numpy as np
import pandas as pd

import gpytorch

def read_data(dataset, only_scale = False, **kwargs):
    D = loadmat("./data/{}.mat".format(dataset))
    data = np.array(D['data'])
    # print(np.mean(data[:, :-1], 0))
    # print(np.std(data[:, :-1], 0))
    # exit()
    train_x, test_x, train_y, test_y = train_test_split(data[:, :-1], data[:, -1], test_size=0.10, random_state=np.random.randint(10000))
    train_x = torch.tensor(train_x)
    train_y = torch.tensor(train_y)
    test_x = torch.tensor(test_x)
    test_y = torch.tensor(test_y)

    y_std_train = torch.std(train_y)
    y_std_full = torch.std(torch.cat([train_y, test_y]))

    kernel = None
    if only_scale:
        return y_mean, y_std
    else:
        return train_x, train_y, test_x, test_y, y_std_full, y_std_train, kernel
