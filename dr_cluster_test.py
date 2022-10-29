import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import umap
from helpers.helperFunctions import *
import plotly.express as px
from sklearn.mixture import GaussianMixture
from sklearn.feature_selection import VarianceThreshold

# loading
df = pd.read_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/events_CN_UMAP.csv',
                 sep=",",
                 encoding='unicode_escape')

# saving IDs
df_id = df[['playerId', 'seasonId', 'teamId', 'map_group', 'pos_group']]
df = df.drop(['playerId', 'seasonId', 'teamId', 'map_group', 'pos_group',], axis=1)

# variance thresholding
vt = VarianceThreshold(threshold=0.003)
_ = vt.fit(df)
mask = vt.get_support() # getting boolean mask
df = df.loc[:, mask] # subsetting the data

outliers = find_outliers_IQR(df)
check = outliers.describe()
df = df.drop(['Shot_zone', 'Offside_zone', 'shot_distance',
              'Simple pass_zone', 'Clearance_zone', 'Air duel_zone', 'Cross_zone', 'Ground attacking duel_zone', 'Head pass_zone', 'Ground loose ball duel_zone', 'Ground defending duel_zone', 'High pass_zone', 'Smart pass_zone', 'Foul_zone', 'Acceleration_zone'], axis=1)

# possession and non-posession sets
df_ip = df.iloc[:, np.r_[0:4, 5:8, 10:19, 23:27, 29:31, 36:44, 46, 48, 51:57]]
df_op = df.iloc[:, np.r_[4, 8:10, 19:23, 27:29, 31:34, 44:46, 47, 49:51]]

# applying UMAP - remember to install pynndescent to make it run faster
dr_ip = umap.UMAP(n_neighbors=50, min_dist=0.0, n_components=2, random_state=42).fit_transform(df_ip)
dr_op = umap.UMAP(n_neighbors=50, min_dist=0.0, n_components=2, random_state=42).fit_transform(df_op)

# when n_components=2
dr2 = pd.DataFrame(dr_op, columns=["x", "y"])
dr2 = df_id.join(dr2)
plot = sns.scatterplot(data=dr2, x="x", y="y", hue = "map_group")
plt.show()

# optimal GMM model
opt_clus(dr_ip)
opt_clus(dr_op)

# clustering, in-possessions
gmm_ip = GaussianMixture(n_components=17, covariance_type='full', random_state=42).fit(dr_ip).predict(dr_ip)
plt.scatter(dr_ip[:, 0], dr_ip[:, 1], c=gmm_ip, s=40, cmap='viridis')
plt.show()

# clustering, out-of-possessions
gmm_op = GaussianMixture(n_components=5, covariance_type='full', random_state=42).fit(dr_op).predict(dr_op)
plt.scatter(dr_op[:, 0], dr_op[:, 1], c=gmm_op, s=40, cmap='viridis')
plt.show()

# merging
df2_ip = pd.concat([df_ip.reset_index(drop=True),gmm_to_df(gmm_ip, "ip").reset_index(drop=True)], axis=1)
df2_id = pd.concat([df_id.reset_index(drop=True),gmm_to_df(gmm_ip, "ip").reset_index(drop=True)], axis=1)

df2_id.groupby(['ip_cluster'], as_index=False).count()

# validation
val = pd.read_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/val.csv', sep=";", encoding='unicode_escape')
val_df = pd.merge(val, df2_id, on='playerId')
val_df.to_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/valDF.csv', index=False)


# ---------------------------------------------------------------------------------



# creating DFs
c0 = df2_ip[df2_ip.ip_cluster == 0]
c1 = df2_ip[df2_ip.ip_cluster == 1]
c2 = df2_ip[df2_ip.ip_cluster == 2]
c3 = df2_ip[df2_ip.ip_cluster == 3]

# dropping cluster
c0 = c0.drop(['ip_cluster'], axis=1)
c1 = c1.drop(['ip_cluster'], axis=1)
c2 = c2.drop(['ip_cluster'], axis=1)
c3 = c3.drop(['ip_cluster'], axis=1)

# creating ID DFs
c0_id = df2_id[df2_id.ip_cluster == 0]
c1_id = df2_id[df2_id.ip_cluster == 1]
c2_id = df2_id[df2_id.ip_cluster == 2]
c3_id = df2_id[df2_id.ip_cluster == 3]

# dropping cluster
c0_id = c0_id.drop(['ip_cluster'], axis=1)
c1_id = c1_id.drop(['ip_cluster'], axis=1)
c2_id = c2_id.drop(['ip_cluster'], axis=1)
c3_id = c3_id.drop(['ip_cluster'], axis=1)

# applying UMAP - remember to install pynndescent to make it run faster
dr0_ip = umap.UMAP(n_neighbors=50, min_dist=0.0, n_components=2, random_state=42).fit_transform(c0)
dr1_ip = umap.UMAP(n_neighbors=50, min_dist=0.0, n_components=2, random_state=42).fit_transform(c1)
dr2_ip = umap.UMAP(n_neighbors=50, min_dist=0.0, n_components=2, random_state=42).fit_transform(c2)
dr3_ip = umap.UMAP(n_neighbors=50, min_dist=0.0, n_components=2, random_state=42).fit_transform(c3)

# when n_components=2
test = pd.DataFrame(dr0_ip, columns=["x", "y"])
test = c0_id.join(test)
plot = sns.scatterplot(data=test, x="x", y="y", hue = "pos_group")
plt.show()
test.groupby(['map_group'], as_index=False).count()

# creating clusters
opt_clus(dr0_ip) # running for 0-4
gmm0_ip = GaussianMixture(n_components=6, covariance_type='full', random_state=42).fit(dr0_ip).predict(dr0_ip)
gmm1_ip = GaussianMixture(n_components=5, covariance_type='full', random_state=42).fit(dr1_ip).predict(dr1_ip)
gmm2_ip = GaussianMixture(n_components=5, covariance_type='full', random_state=42).fit(dr2_ip).predict(dr2_ip)
gmm3_ip = GaussianMixture(n_components=5, covariance_type='full', random_state=42).fit(dr3_ip).predict(dr3_ip)

# visualizing to check
plt.scatter(dr0_ip[:, 0], dr0_ip[:, 1], c=gmm0_ip, s=40, cmap='viridis')
plt.show()

# merging into df
c0_id = pd.concat([c0_id.reset_index(drop=True),gmm_to_df(gmm0_ip, "ip").reset_index(drop=True)], axis=1)
c1_id = pd.concat([c1_id.reset_index(drop=True),gmm_to_df(gmm1_ip, "ip").reset_index(drop=True)], axis=1)
c2_id = pd.concat([c2_id.reset_index(drop=True),gmm_to_df(gmm2_ip, "ip").reset_index(drop=True)], axis=1)
c3_id = pd.concat([c3_id.reset_index(drop=True),gmm_to_df(gmm3_ip, "ip").reset_index(drop=True)], axis=1)

# adjusting clust no.
c1_id['ip_cluster'] = c1_id['ip_cluster'] + 6
c2_id['ip_cluster'] = c2_id['ip_cluster'] + 11
c3_id['ip_cluster'] = c3_id['ip_cluster'] + 16

# val
val = pd.read_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/val.csv', sep=";", encoding='unicode_escape')
frames = [c0_id, c1_id, c2_id, c3_id]
val2 = pd.concat(frames)
val_df = pd.merge(val, val2, on='playerId')
val_df.to_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/valDF.csv', index=False)
