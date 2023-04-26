
import streamlit as st
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


# set page title
st.set_page_config(page_title="Drug Overdose Deaths in the USA")

# set up sidebar
st.sidebar.title("Settings")
k = st.sidebar.slider("K", min_value=1, max_value=20, value=8)
scheme = st.sidebar.selectbox("Choropleth Scheme", ["quantiles", "equal_interval", "fisher_jenks"])
basemap = st.sidebar.selectbox("Basemap", ["CartoDB Positron", "CartoDB Dark_Matter", "Stamen Terrain", "Stamen Toner", "OpenStreetMap Mapnik", "OpenTopoMap"])
normalize = st.sidebar.checkbox("Normalize values")

# load data
od_filepath = 'https://raw.githubusercontent.com/prasuj/ML-ASSIGNMENT-3/main/CDC_Injury_Center_Drug_Overdose_Deaths.csv?token=GHSAT0AAAAAACBNGTT7WHBNQEU7LTNDON4IZCI4W7Q'
oddb = pd.read_csv(od_filepath, encoding="ISO-8859-1")

contiguous_usa = gpd.read_file(gplt.datasets.get_path('contiguous_usa'))
result = contiguous_usa.set_index('state').join(oddb.set_index('State'))

# set up figure and a single axis
g, ax = plt.subplots(1, figsize=(9, 9))

# build choropleth
column = '2019 Age-adjusted Rate (per 100,000 population)'
if normalize:
    column += '_std'
result.plot(
    column=column,
    cmap='Blues',
    scheme=scheme,
    k=k,
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
    source=contextily.providers[basemap]
)

# remove axis
ax.set_axis_off();

# generate W from the GeoDataFrame
w = weights.distance.KNN.from_dataframe(result, k=k)
# row-standardization
w.transform = 'R'

result[f'w_{column}'] = weights.spatial_lag.lag_spatial(w, result[column])

if normalize:
    result[column] = (result[column] - result[column].mean()) / result[column].std()
    result[f'w_{column}'] = (result[f'w_{column}'] - result[f'w_{column}'].mean()) / result[f'w_{column}'].std()

# setup the figure and axis
f, ax = plt.subplots(1, figsize=(6,  6))
# plot values
seaborn.regplot(
    x=column, y=f'w_{column}', data=result, ci=None
)

# set plot title and axis labels
ax.set_title('Spatial Autocorrelation')
ax.set_xlabel(column)
ax.set_ylabel(f'Spatial Lag of {column}')

# calculate Moran's I
moran = esda.Moran(result[column], w)

# add Moran's I to the plot
ax.annotate(f'Moran\'s I = {moran.I:.2f}\np-value = {moran.p_sim:.2f}', xy=(0.05, 0.85), xycoords='axes fraction')

# show plot
st.pyplot(f)
