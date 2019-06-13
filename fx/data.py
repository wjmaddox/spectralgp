import math
import torch
import numpy as np

def get_data(dataset, standardize=False):
    from scipy.io import loadmat

    if dataset == 'fx':
        mat = loadmat('./data/fx.mat')
        mat['x'] = mat['x'].astype(np.float)
        mat['xtest'] = mat['xtest'].astype(np.float)

        # standardize data
        y_means = np.nanmean(mat['y'],0)
        y_std = np.nanstd(mat['y'],0)
        
        mat['y'] = (mat['y'] - y_means) / y_std
        mat['ytest'] = (mat['ytest'] - y_means) / y_std
        

        # variables to keep for training set
        # remove the nan variables
        tokeep = (np.isnan(mat['y']) * -1 + 1) > 0
        # note that dimensions 3, 5, 8 are nan in the proper places bc
        # we got the data from https://github.com/trungngv/cogp
        #print(mat['y'][:,[3,5,8]])

        train_x_list = [torch.tensor(mat['x'][tokeep[:,i]]).view(-1) for i in range(mat['y'].shape[1])]
        train_y_list = [torch.tensor(mat['y'][tokeep[:,i],i]) for i in range(mat['y'].shape[1])]

        # variables to keep for testing set
        tokeep_test = (np.isnan(mat['ytest']) * -1 + 1) > 0
        test_x_list = [torch.tensor(mat['xtest'][tokeep_test[:,i]]).view(-1) for i in range(mat['ytest'].shape[1])]
        test_y_list = [torch.tensor(mat['ytest'][tokeep_test[:,i],i]) for i in range(mat['ytest'].shape[1])]

        # note that we only care about variables 3, 5, 8
        return train_x_list, train_y_list, test_x_list, test_y_list

    elif dataset == 'jura':
        df = pd.read_csv('./data/jura.txt', delim_whitespace=True)

        train_df = df.iloc[0:259]
        test_df = df.iloc[259:]

        train_df = (train_df - train_df.mean())/train_df.std()
        test_df = (test_df - test_df.mean())/test_df.std()

        train_x = torch.tensor(train_df[['X','Y']].values).type(torch.float64)
        train_y = torch.tensor(train_df[['Cd','Ni','Zn']].values).type(torch.float64)

        test_x = torch.tensor(test_df[['X','Y']].values).type(torch.float64)
        test_y = torch.tensor(test_df[['Cd','Ni','Zn']].values).type(torch.float64)

        y_std = torch.std(torch.cat([train_y, test_y], dim=0), dim=0)

        kernel = None

        return train_x, train_y, test_x, test_y
