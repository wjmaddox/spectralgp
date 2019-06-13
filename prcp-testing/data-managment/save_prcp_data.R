setwd("~/Google Drive/PRCP-Data/")
load("./master_file.Rda")

prcp = trace_PRCP
stn_name = stn.name
lon_lat = lon.lat

save_file = "~/Google Drive/research/spectrum/spectralgp/prcp-testing/prcp_data.RData"
save("prcp", "stn_name", "lon_lat", "days", "years", file=save_file)

rm(list = ls())
load("./prcp_data.RData")
