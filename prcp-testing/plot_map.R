library(ggplot2)
library(fields)
setwd("~/Documents/GitHub/spectralgp/prcp-testing/")
load("./data-managment/prcp_data.RData")

VT = 1082:1090 + 1
MA = 403:413 + 1
NH = 631:636 + 1
ME = 374:384 + 1
RI = 921:923 + 1
CT = 133:136 + 1
NY = 677:733 + 1
NJ = 636:647 + 1
all = c(VT, MA, NH, ME, RI, CT, NY, NJ)

plot_df = data.frame(lon_lat[all, ], rep(1, length(all)))
colnames(plot_df) = c("lon", "lat", "Trace")
plot_range = range(plot_df$Trace)

us = map_data("state")
us = subset(us, region %in% c("maine", "new hampshire", "vermont", "massachusetts", "rhode island", "connecticut",
                              "new york", "new jersey"))
plt = ggplot(plot_df, aes(x = lon, y = lat)) + theme_minimal() +
  geom_polygon(data = us, aes(x = long,y = lat, group = group),
                   fill = "grey", color = "black") +
  geom_point(size = 5, aes(color = Trace)) +
  scale_color_gradientn(guide=FALSE, colors="steelblue") +
      ggtitle("") + ## CHANGE THIS ##
      theme(plot.title = element_text(hjust = 0.5)) + 
        coord_cartesian(xlim = c(-80, -67), ylim = c(39, 47.5)) + 
    xlab("Longitude") + ylab("Latitude") +theme(axis.text=element_text(size=12),
                                                axis.title=element_text(size=14,face="bold"))
  ## CHANGE THESE ##

plot(plt)

