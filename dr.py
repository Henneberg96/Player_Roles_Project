import numpy as np
import pandas as pd
import umap
# from umap import UMAP
import umap.umap_ as umap
from sklearn.datasets import load_digits

df = pd.read_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/events_CN.csv',
                 sep=",",
                 encoding='unicode_escape')

reducer = umap.UMAP()

test = umap.UMAP(n_neighbors=5,
                      min_dist=0.3,
                      metric='correlation').fit_transform(df)
