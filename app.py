import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import contextily
import seaborn
from libpysal import weights
import streamlit as st

# Load the shapefile containing the state boundaries

contiguous_usa = pd.read_csv('output.csv')

# Load the drug overdose data
overdoses = pd.read_csv('CDC_Injury_Center_Drug_Overdose_Deaths.csv', encoding="ISO-8859-1")

# Join the overdose data with the state boundaries based on state name
result = contiguous_usa.set_index('state').join(overdoses.set_index('State'))

# set up the Streamlit app
st.title('US Drug Overdose Rates')
st.write('Visualization of US drug overdose rates by state')

# create a slider for k
k = st.slider('Select k for KNN weights', min_value=1, max_value=20, value=8, step=1)

# generate W from the GeoDataFrame
w = weights.distance.KNN.from_dataframe(result, k=k)
# row-standardization
w.transform = 'R'

result['w_2019 Age-adjusted Rate (per 100,000 population)'] = weights.spatial_lag.lag_spatial(w, result['2019 Age-adjusted Rate (per 100,000 population)'])

result['2019 Age-adjusted Rate (per 100,000 population)_std'] = (result['2019 Age-adjusted Rate (per 100,000 population)'] - result['2019 Age-adjusted Rate (per 100,000 population)'].mean())
result['w_2019 Age-adjusted Rate (per 100,000 population)_std'] = (result['w_2019 Age-adjusted Rate (per 100,000 population)'] - result['2019 Age-adjusted Rate (per 100,000 population)'].mean())

# setup the figure and axis
fig, ax = plt.subplots(1, figsize=(6, 6))
# plot values
seaborn.regplot(x='2019 Age-adjusted Rate (per 100,000 population)_std', y='w_2019 Age-adjusted Rate (per 100,000 population)_std', data=result, ci=None)
plt.title('Spatial Autocorrelation for US Drug Overdose Rates')
plt.xlabel('Age-adjusted Rate (per 100,000 population)')
plt.ylabel('Spatial Lag of Age-adjusted Rate (per 100,000 population)')

# set up figure and a single axis
fig, ax = plt.subplots(1, figsize=(9, 9))

# build choropleth
result.plot(column='2019 Age-adjusted Rate (per 100,000 population)',
            cmap='Blues',
            scheme='quantiles',
            k=5,
            edgecolor='0.8',
            linewidth=0.8,
            alpha=0.75,
            legend=True,
            legend_kwds=dict(loc=2),
            ax=ax)

# add basemap
contextily.add_basemap(ax,
                        crs=result.crs,
                        source=contextily.providers.CartoDB.VoyagerNoLabels)

# remove axis
ax.set_axis_off()

# show the plots in Streamlit
st.pyplot(fig)
st.pyplot(seaborn) # Show seaborn plot
