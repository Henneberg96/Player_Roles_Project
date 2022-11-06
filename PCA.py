import numpy as np
from sklearn.decomposition import PCA
from helpers.helperFunctions import *
import numba
import umap
import pandas as pd
import matplotlib.pyplot as plt


# Load dataset
raw = pd.read_csv("C:/ITU/ITU_Research_Project/clustered_data/events_CN_UMAP.csv", sep = ",", encoding='unicode_escape')
raw = pd.read_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/events_CN_UMAP.csv',
                 sep=",", encoding='unicode_escape')

# loading
raw = pd.read_csv('C:/ITU/ITU_Research_Project/preprocessed/events_CN.csv',
                  sep=",", encoding='unicode_escape')

# saving IDs
df_id = raw[['playerId', 'seasonId', 'map_group', 'pos_group']]
raw = raw.drop(['playerId', 'seasonId', 'map_group', 'pos_group', ], axis=1)

#Create pca instance
pca = PCA(n_components=0.8)

#Split data in train and test sets
trans = pca.fit_transform(raw)

# Compute an array of principal components
explained_variance = pca.explained_variance_ratio_

#extract number of principal components
print(len(explained_variance))
print(sum(explained_variance))
print(explained_variance)

plt.title("Line graph")
plt.plot(explained_variance, color ="green", marker='x')
plt.show()

opt_clus(trans)

# testing PCA + UMAP
dr = umap.UMAP(n_neighbors=50, min_dist=0.0, n_components=2, random_state=42).fit_transform(trans)
opt_clus(dr)