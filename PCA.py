import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from advanced_pca import CustomPCA


# Load dataset
raw = pd.read_csv("C:/ITU/ITU_Research_Project/preprocessed/events_CN.csv", sep = ",", encoding='unicode_escape')

# loading
raw = pd.read_csv('C:/ITU/ITU_Research_Project/preprocessed/events_CN.csv',
                  sep=",", encoding='unicode_escape')

# saving IDs
df_id = raw[['playerId', 'seasonId', 'teamId', 'map_group', 'pos_group']]
raw = raw.drop(['playerId', 'seasonId', 'teamId', 'map_group', 'pos_group', ], axis=1)

# variance thresholding
vt = VarianceThreshold(threshold=0.003)
_ = vt.fit(raw)
mask = vt.get_support() # getting boolean mask
raw = raw.loc[:, mask] # subsetting the data

# possession and non-posession sets
df_op = raw.iloc[:, np.r_[4, 8:10, 19:23, 27:29, 31:34, 44:46, 47, 49:51]]
df_ip = raw.iloc[:, np.r_[0:4, 5:8, 10:19, 23:27, 29:31, 36:44, 46, 48, 51:57]]


#removal of redundant columns
data = raw.drop(['playerId', 'seasonId', 'map_group', 'pos_group', 'teamId'], axis=1)

#Split in train and test set
def_input_train, def_input_test = train_test_split(df_op,test_size=0.20, random_state=42)

#Create pca instance
pca = PCA(n_components=0.8)

#Split data in train and test sets
X_train = pca.fit_transform(def_input_train)
X_test = pca.transform(def_input_test)


# Compute an array of principal components
explained_variance = pca.explained_variance_ratio_

#extract number of principal components
print(len(explained_variance))
print(sum(explained_variance))
print(explained_variance)


plt.title("Line graph")
plt.plot(explained_variance, color ="green", marker='x')
plt.show()

pd.DataFrame(data[data.columns].values)

pd.DataFrame(data[data.columns].values)


data[data.columns] = pd.DataFrame(data[data.columns].values, columns=data.columns, index=data.index) ##scale values inplace
X = data[data.columns].values

vpca_Xs = CustomPCA(n_components=8, rotation='varimax').fit_transform(X)


varimax_pca5 = CustomPCA(n_components=8, rotation='varimax').fit(pd.DataFrame(data.values))
