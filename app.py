import seaborn as sns
import numpy as np # linear algebra
import matplotlib.pyplot as plt   # Graphics
from matplotlib import colors
import seaborn                    # Graphics
import geopandas as gpd           # Spatial data manipulation
import pandas as pd               # Tabular data manipulation
import xarray                     # Surface data manipulation
from pysal.explore import esda    # Exploratory spatial analytics
from pysal.lib import weights     # Spatial weights
import contextily                 # Background tiles
import geoplot as gplt
import geoplot.crs as gcrs
import streamlit as st

# save filepath to variable for easier access
od_filepath = 'https://raw.githubusercontent.com/prasuj/ML-ASSIGNMENT-3/main/CDC_Injury_Center_Drug_Overdose_Deaths.csv?token=GHSAT0AAAAAACBNGTT7WHBNQEU7LTNDON4IZCI4W7Q'
# read the data and store data in dataframe
# choosing to save as pandas, not geopandas
# adding encoding to prevent a 'UnicodeDecodeError'
oddb = pd.read_csv(od_filepath, encoding = "ISO-8859-1")


# finding data with state shapes and joining it
contiguous_usa = gpd.read_file(gplt.datasets.get_path('contiguous_usa'))
result = contiguous_usa.set_index('state').join(oddb.set_index('State'))
# set up figure and a single axis
g, ax = plt.subplots(1, figsize=(9, 9))

# build choropleth
result.plot(
    column='2019 Age-adjusted Rate (per 100,000 population)',
    cmap='Blues',
    scheme='quantiles',
    k=5,
    edgecolor='0.8',
    linewidth=0.8,
    alpha=0.75,
    legend=True,
    legend_kwds=dict(loc=2),
    ax=ax
)

# add basemap
contextily.add_basemap(
    ax,
    crs=result.crs,
    source=contextily.providers.CartoDB.VoyagerNoLabels
)

# remove axis
ax.set_axis_off();

# generate W from the GeoDataFrame
w = weights.distance.KNN.from_dataframe(result, k=8)
# row-standardization
w.transform = 'R'


result['w_2019 Age-adjusted Rate (per 100,000 population)'] = weights.spatial_lag.lag_spatial(w, result['2019 Age-adjusted Rate (per 100,000 population)'])
result['2019 Age-adjusted Rate (per 100,000 population)_std'] = ( result['2019 Age-adjusted Rate (per 100,000 population)'] - result['2019 Age-adjusted Rate (per 100,000 population)'].mean() )
result['w_2019 Age-adjusted Rate (per 100,000 population)_std'] = ( result['w_2019 Age-adjusted Rate (per 100,000 population)'] - result['2019 Age-adjusted Rate (per 100,000 population)'].mean() )

# create a slider to adjust k value
k = st.slider('Select k value:', 2, 10, 8, 1)

# update the W matrix based on the slider value
w = weights.distance.KNN.from_dataframe(result, k=k)
w.transform = 'R'

# setup the figure and axis
f, ax = plt.subplots(1, figsize=(6,  6))
# plot values
seaborn.regplot(
    x='2019 Age-adjusted Rate (per 100,000 population)_std',
    y='w_2019 Age-adjusted Rate (per 100,000 population)_std',
    data=result,
    ax=ax,
    line_kws={'color': 'red'},
    scatter_kws={'alpha': 0.5}
)
# set x and y axis labels
ax.set_xlabel('2019 Age-adjusted Rate (per 100,000 population)')
ax.set_ylabel('Spatial Lag of 2019 Age-adjusted Rate (per 100,000 population)')

# set plot title
ax.set_title(f'Spatial Lag plot using {k} neighbors')

# remove borders
sns.despine(ax=ax, left=True, bottom=True)
