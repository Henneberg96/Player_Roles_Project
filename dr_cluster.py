import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import umap
from helpers.helperFunctions import *
import plotly.express as px
from sklearn import mixture
from scipy import linalg
import matplotlib as mpl
import itertools

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

# getting IDs for map visualization
df_id_def = df_id[df_id.pos_group == 'DEF']
df_id_mid = df_id[df_id.pos_group == 'MID']
df_id_wing = df_id[df_id.pos_group == 'WING']
df_id_att = df_id[df_id.pos_group == 'ATT']

# applying UMAP - remember to install pynndescent to make it run faster
dr_def = umap.UMAP(n_neighbors=100, min_dist=0.0, n_components=2, random_state=42).fit_transform(df_def)
dr_mid = umap.UMAP(n_neighbors=100, min_dist=0.0, n_components=2, random_state=42).fit_transform(df_mid)
dr_wing = umap.UMAP(n_neighbors=100, min_dist=0.0, n_components=2, random_state=42).fit_transform(df_wing)
dr_att = umap.UMAP(n_neighbors=100, min_dist=0.0, n_components=2, random_state=42).fit_transform(df_att)

# when n_components=2
dr2 = pd.DataFrame(dr_att, columns=["x", "y"])
dr2 = df_id_att.join(dr2)
plot = sns.scatterplot(data=dr2, x="x", y="y", hue = "map_group")
plt.show()

# when n_components=3
dr2 = pd.DataFrame(dr_att, columns=["x", "y", "z"])
dr2 = pd.merge(df_id_def, dr2, left_index=True, right_index=True)
fig = px.scatter_3d(dr2, x='x', y='y', z='z', color='map_group')
fig.show()

# optimal GMM model
opt_clus(dr_def)
opt_clus(dr_mid)
opt_clus(dr_wing)
opt_clus(dr_att)

# visualizing defenders
gmm_def = mixture.GaussianMixture(n_components=8, covariance_type='full', random_state=42).fit(dr_def).predict(dr_def)
plt.scatter(dr_def[:, 0], dr_def[:, 1], c=gmm_def, s=40, cmap='viridis')
plt.show()

# visualizing midfielders
gmm_mid = mixture.GaussianMixture(n_components=8, covariance_type='full', random_state=42).fit(dr_mid).predict(dr_mid)
plt.scatter(dr_mid[:, 0], dr_mid[:, 1], c=gmm_mid, s=40, cmap='viridis')
plt.show()

# visualizing wingers
gmm_wing = mixture.GaussianMixture(n_components=3, covariance_type='full', random_state=42).fit(dr_wing).predict(dr_wing)
plt.scatter(dr_wing[:, 0], dr_wing[:, 1], c=gmm_wing, s=40, cmap='viridis')
plt.show()

# visualizing attackers
gmm_att = mixture.GaussianMixture(n_components=5, covariance_type='full', random_state=42).fit(dr_att).predict(dr_att)
plt.scatter(dr_att[:, 0], dr_att[:, 1], c=gmm_att, s=40, cmap='viridis')
plt.show()

# merging into df
df_id_def = df_id_def.join(gmm_to_df(gmm_def))
df_id_mid = df_id_mid.join(gmm_to_df(gmm_mid))
df_id_wing = df_id_wing.join(gmm_to_df(gmm_wing))
df_id_att = df_id_att.join(gmm_to_df(gmm_att))

# visualization to validate merge
test = pd.DataFrame(dr_wing, columns=["x", "y"])
test = df_id_wing.join(test)
plot = sns.scatterplot(data=test, x="x", y="y", hue = "cluster")
plt.show()

# exporting clusters

