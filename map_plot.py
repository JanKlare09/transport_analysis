import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors
import mapclassify


# get coefficient surface w_sh
legend_classes = mapclassify.NaturalBreaks(gwr_map_2.loc[(~gwr_map_2['w_sh'].isna()), 'w_sh'], k = 5)
    
legend_classes = mapclassify.UserDefined(gwr_map_2.loc[(~gwr_map_2['w_sh'].isna()), 'w_sh'], bins = [-5.55, 0, 5.64, 8.93, 12.69])
mapping = dict([(i,s) for i,s in enumerate(legend_classes.get_legend_classes())])

# plot
fig = plt.figure(figsize = (10,8))
ax = fig.add_axes([0,0,0.95,0.95])
gwr_map_2.loc[(~gwr_map_2['w_sh'].isna())].assign(cl = legend_classes.yb).plot(column = 'cl', categorical = True,\
                                             k = 6, cmap = 'coolwarm', norm = matplotlib.colors.Normalize(-1.8,3.5),\
                                             linewidth = 0.3,\
                                            alpha = 0.7,\
                                            ax = ax, edgecolor = 'black', legend = True,\
                                            legend_kwds = {'loc':'lower left',\
                                                           'fontsize': 10,\
                                                            'frameon': False,\
                                                            'title': 'GWR coefficients\nw_sh\n',\
                                                            'title_fontsize': 12,\
                                                             'edgecolor': 'gray'})
gwr_map_2.loc[(gwr_map_2['w_sh'].isna())].plot(ax = ax, color = 'white',\
                                               edgecolor = 'lightgrey',\
                                            hatch = 'oo', linewidth = 0.3,\
                                            alpha = 0.85)
excl_zones.plot(ax = ax, color = 'white', edgecolor = 'gray', hatch = '/////',\
                linewidth = 0.3)
la_shp.plot(ax = ax, color = 'none', edgecolor = 'black', linewidth = 1.5,\
            alpha = 0.7)
ax.text(0.27,0.026, 'HW zones excluded from the analysis', horizontalalignment = 'left',\
        verticalalignment = 'center', transform = ax.transAxes)
ax.text(0.229,0.027, '/////', horizontalalignment = 'left',\
        verticalalignment = 'center', transform = ax.transAxes, fontsize = 10,\
            color = 'gray', alpha = 0.95, fontweight = 'bold', fontstyle = 'italic')                                                       
ax.text(0.27,0.057, 'Coefficient not significant', horizontalalignment = 'left',\
        verticalalignment = 'center', transform = ax.transAxes)    
ax.text(0.229,0.058, 'o o o', horizontalalignment = 'left',\
        verticalalignment = 'center', transform = ax.transAxes, fontsize = 9,\
            color = 'lightgrey', fontweight = 'black')
ax.text(0.229,0.058, 'o o o', horizontalalignment = 'left',\
        verticalalignment = 'center', transform = ax.transAxes, fontsize = 9,\
            color = 'lightgrey', fontweight = 'black')
ax.text(0.229,0.058, 'o o o', horizontalalignment = 'left',\
        verticalalignment = 'center', transform = ax.transAxes, fontsize = 9,\
            color = 'lightgrey', fontweight = 'black')     
ax.set_axis_off()
replace_legend_items(ax.get_legend(), mapping)
plt.savefig('path/mod_2_w_sh.png',\
            dpi = 300)