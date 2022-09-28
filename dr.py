import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import umap
from helpers.helperFunctions import *
import plotly.express as px

df = pd.read_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/events_CN.csv',
                 sep=",",
                 encoding='unicode_escape')

test = pd.DataFrame(df.isna().sum())

#saving IDs and merging with positions
df2 = df[['playerId', 'seasonId']]
pos = pd.read_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/Wyscout_Positions_Minutes.csv', sep=";", encoding='unicode_escape')
pos2 = pd.read_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/Wyscout_Simple_Positions.csv', sep=";", encoding='unicode_escape')
pos2.drop(columns=pos2.columns[0], axis=1, inplace=True)
pos2 = pos2.drop(['pos_group', 'radar_group'], axis=1)
pos = pd.merge(pos, pos2, on=['position'])
pos.drop(columns=pos.columns[0], axis=1, inplace=True)
pos = pos.drop(['matchId', 'teamId', 'position', 'time'], axis=1)

pos = pos.groupby(['playerId', 'seasonId'], as_index=False).agg(gmodeHelp)
df = pd.merge(df, pos, on=['playerId', 'seasonId'])
df2 = pd.merge(df2, pos, on=['playerId', 'seasonId'])

# removing goalkeepers
df_o = df[df.map_group != 'GK']
nan_check = pd.DataFrame(df_o.isna().sum())

df_o_pos = df_o[['playerId', 'seasonId', "map_group"]]
df_o = df_o.iloc[:, np.r_[0:88, 89:92, 94:97, 99:101, 104:106, 107:109, 111, 119:127, 128:137, 138:141, 143:153]]
df_o = df_o.fillna(0)

# applying UMAP - remember to install pynndescent to make it run faster
dr = umap.UMAP(n_neighbors=80, min_dist=0.1, n_components=2, random_state=42).fit_transform(df_o)

# when n_components=2
dr2 = pd.DataFrame(dr, columns=["x", "y"])
dr2 = pd.merge(df_o_pos, dr2, left_index=True, right_index=True)
dr2.insert(3, 'pos_group', dr2.apply(lambda row: pos_group(row), axis=1), allow_duplicates=True)
plot = sns.scatterplot(data=dr2, x="x", y="y", hue = "pos_group")
plt.show()

# when n_components=3
dr2 = pd.DataFrame(dr, columns=["x", "y", "z"])
dr2 = pd.merge(df_o_pos, dr2, left_index=True, right_index=True)
dr2.insert(3, 'pos_group', dr2.apply(lambda row: pos_group(row), axis=1), allow_duplicates=True)
fig = px.scatter_3d(dr2, x='x', y='y', z='z', color='pos_group')
fig.show()
