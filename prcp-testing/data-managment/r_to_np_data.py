import rpy2.robjects as robjects
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("..")

robjects.r['load']("prcp_data.RData")
print(robjects.r['ls']())
stn_name = np.array(robjects.r['stn_name'])
lon_lat = np.array(robjects.r['lon_lat'])
prcp = np.array(robjects.r['prcp'])
days = np.array(robjects.r['days'])
years = np.array(robjects.r['years'])

fname = "../prcp_data"
np.savez(fname, stn_name=stn_name, lon_lat=lon_lat, prcp=prcp, days=days, years=years)

# out = np.load("/Users/greg/Google Drive/research/spectrum/spectralgp/prcp-testing/prcp_data.npz")
