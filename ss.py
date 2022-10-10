import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from helpers.helperFunctions import *
from helpers.test import *
import time
from sklearn import metrics
from sklearn.metrics import pairwise_distances

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

# You et al., Scalable Sparse Subspace Clustering by Orthogonal Matching Pursuit, CVPR 2016
model_ensc = ElasticNetSubspaceClustering(n_clusters=5,affinity='nearest_neighbors',algorithm='spams',active_support=True,gamma=200,tau=0.9)
model_ssc_omp = SparseSubspaceClusteringOMP(n_clusters=5,affinity='symmetrize',n_nonzero=5,thr=1.0e-5)

clustering_algorithms = (
    ('EnSC via active support solver', model_ensc),
    ('SSC-OMP', model_ssc_omp),
)

for name, algorithm in clustering_algorithms:
    t_begin = time.time()
    algorithm.fit(df_att)
    t_end = time.time()
    s_score = metrics.silhouette_score(df_att, algorithm.labels_, metric='euclidean')

    print('Algorithm: {}. silhoutte_score: {}, Running time: {}'.format(name, s_score, t_end - t_begin))


model_ssc_omp.fit(df_att)
print(metrics.silhouette_score(df_att, model_ensc.labels_, metric='euclidean'))