import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import umap
from helpers.helperFunctions import *
import plotly.express as px

# loading
df = pd.read_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/events_CN.csv',
                 sep=",",
                 encoding='unicode_escape')

# creating data set
df_id = df[['playerId', 'seasonId', 'map_group', 'pos_group']]
df_def = df[df.pos_group == 'DEF']
df_mid = df[df.pos_group == 'MID']
df_wing = df[df.pos_group == 'WING']
df_att = df[df.pos_group == 'ATT']

# dropping the ids
df_def = df_def.drop(['playerId', 'seasonId', 'map_group', 'pos_group'], axis=1)
df_mid = df_mid.drop(['playerId', 'seasonId', 'map_group', 'pos_group'], axis=1)
df_wing = df_wing.drop(['playerId', 'seasonId', 'map_group', 'pos_group'], axis=1)
df_att = df_att.drop(['playerId', 'seasonId', 'map_group', 'pos_group'], axis=1)

# applying UMAP - remember to install pynndescent to make it run faster
dr = umap.UMAP(n_neighbors=100, min_dist=0.0, n_components=2, random_state=42).fit_transform(df_att)

# getting IDs for map visualization
df_id_def = df_id[df_id.pos_group == 'DEF']
df_id_mid = df_id[df_id.pos_group == 'MID']
df_id_wing = df_id[df_id.pos_group == 'WING']
df_id_att = df_id[df_id.pos_group == 'ATT']

# when n_components=2
dr2 = pd.DataFrame(dr, columns=["x", "y"])
dr2 = pd.merge(df_id_att, dr2, left_index=True, right_index=True)
plot = sns.scatterplot(data=dr2, x="x", y="y", hue = "map_group")
plt.show()

# when n_components=3
dr2 = pd.DataFrame(dr, columns=["x", "y", "z"])
dr2 = pd.merge(df_id_def, dr2, left_index=True, right_index=True)
fig = px.scatter_3d(dr2, x='x', y='y', z='z', color='map_group')
fig.show()
