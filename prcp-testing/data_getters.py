import math
import torch
import numpy as np
import sys
sys.path.append("..")

def get_pos_prcp_time_series(stn, prcp, days, years, sample_window):
    ts_out = np.zeros(365)
    for day in range(365):
        # set up subsample of days #
        keep_days = [i % 365 for i in range(day - sample_window, day + sample_window + 1)]
        if 0 in keep_days:
            keep_days[keep_days.index(0)] = 365

        # get out data
        day_data = prcp[np.isin(days, keep_days), stn] # get out all stn obs for this day
        day_data[np.isnan(day_data)] = 0 # remove nans
        ts_out[day] = np.mean(day_data[day_data > 0]) # avg positive observations

    return ts_out

def get_data(train_list, test_list, sample_window=21, standardize=False):
    ## read in data from files ##
    dat = np.load("prcp_data.npz")
    stn_name = dat['stn_name']
    lon_lat = dat['lon_lat']
    prcp = dat['prcp']
    days = dat['days']
    years = dat['years']

    ## get out requested data ##
    n_train = len(train_list)
    train_out = np.zeros((n_train, 365))
    train_lonlat = np.zeros((n_train, 2))
    train_name = []
    for inder, stn in enumerate(train_list):
        ts = get_pos_prcp_time_series(stn, prcp, days, years, sample_window)
        if standardize:
            y_mean = ts.mean()
            y_std = ts.std()
            # ts = (ts - y_mean)/y_std
            ts = (ts - y_mean)

        train_out[inder, :] = ts
        train_lonlat[inder, :] = lon_lat[stn, :]
        train_name.append(stn_name[stn])

    n_test = len(test_list)
    test_out = np.zeros((n_test, 365))
    test_lonlat = np.zeros((n_test, 2))
    test_name = []
    for inder, stn in enumerate(test_list):
        ts = get_pos_prcp_time_series(stn, prcp, days, years, sample_window)
        if standardize:
            y_mean = ts.mean()
            y_std = ts.std()
            # ts = (ts - y_mean)/y_std
            ts = (ts - y_mean)

        test_out[inder, :] = ts
        test_lonlat[inder, :] = lon_lat[stn, :]
        test_name.append(stn_name[stn])

    days = np.arange(1, 366)
    # if standardize:
    #     dmean = days.mean()
    #     dstd = days.std()
    #     days = (days - dmean)/dstd
    #     days = (days - dmean)

    return [train_out, train_lonlat, train_name,
            test_out, test_lonlat, test_name,
            days]


def get_extrapolation_data(train_list, test_cutoff=300, sample_window=21, standardize=True):
    ## read in data from files ##
    dat = np.load("prcp_data.npz")
    stn_name = dat['stn_name']
    lon_lat = dat['lon_lat']
    prcp = dat['prcp']
    days = dat['days']
    years = dat['years']
    n_stn = len(train_list)

    yr_days = np.arange(1, 366)
    train_days = yr_days[:test_cutoff]
    test_days = yr_days[test_cutoff:]
    n_train = len(train_days)
    n_test = len(test_days)

    ## get out requested data ##
    train_out = np.zeros((n_stn, n_train))
    test_out = np.zeros((n_stn, n_test))
    out_lonlat = np.zeros((n_stn, 2))
    out_name = []
    for inder, stn in enumerate(train_list):
        ts = get_pos_prcp_time_series(stn, prcp, days, years, sample_window)
        if standardize:
            y_mean = ts.mean()
            y_std = ts.std()
            # ts = (ts - y_mean)/y_std
            ts = (ts - y_mean)

        train_out[inder, :] = ts[:test_cutoff]
        test_out[inder, :] = ts[test_cutoff:]
        out_lonlat[inder, :] = lon_lat[stn, :]
        out_name.append(stn_name[stn])


    return [train_days, train_out, test_days, test_out, out_lonlat, out_name]
