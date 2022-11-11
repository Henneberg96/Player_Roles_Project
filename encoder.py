import keras.optimizers
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
import umap
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from helpers.helperFunctions import *
from sklearn.model_selection import train_test_split
import plotly.express as px

# Load data
event_data = pd.read_csv('C:/ITU/ITU_Research_Project/clustered_data/events_CN_UMAP.csv',
                 sep=",",
                 encoding='unicode_escape')
#Remove ids before endoding proces
event_data_cleaned = event_data.drop(['playerId', 'seasonId', 'map_group', 'pos_group', ], axis=1)

#split in train and test data
train, test = train_test_split(event_data_cleaned,  test_size=0.33, random_state=42)

#Endocing dimensions are 24
encoding_dim = 24

#Construction of input layer
input_layer = Input(shape=(encoding_dim,))

#Build model using Dense layers
encoder = Dense(24, activation='relu')(input_layer)
encoder = Dense(20, activation='relu')(encoder)
encoder = Dense(16, activation='relu')(encoder)
encoder = Dense(12, activation='relu')(encoder)
bottleNeck_layer = Dense(10, activation='relu', name='bottleneck_layer')(encoder)

decoder = Dense(10, activation='relu')(bottleNeck_layer)
decoder = Dense(14, activation='relu')(decoder)
decoder = Dense(16, activation='relu')(decoder)
decoder = Dense(20, activation='relu')(decoder)
decoded = Dense(24, activation='sigmoid')(decoder)


#Make model using input og final decoder layer
encoder = Model(input_layer, decoded)
#Compile encoder
encoder.compile(optimizer=keras.optimizers.Adam(learning_rate=0.1), loss='binary_crossentropy',  metrics=['accuracy'])
encoder.summary()
encoded_features = encoder.predict(event_data_cleaned)
encoder.fit(train, train)
bottleneck_output = encoder.get_layer('bottleneck_layer').output
model_bottleneck = Model(inputs = encoder.input, outputs = bottleneck_output)
bottleneck_predictions = model_bottleneck.predict(test)

df = pd.DataFrame(bottleneck_predictions)
df_id = event_data[['playerId', 'seasonId', 'map_group', 'pos_group']]


dr = umap.UMAP(n_neighbors=80, min_dist=0.0, n_components=3, random_state=42).fit_transform(df)
dr2 = pd.DataFrame(dr, columns=["x", "y", "z"])
dr2 = pd.merge(df_id, dr2, left_index=True, right_index=True)
fig = px.scatter_3d(dr2, x='x', y='y', z='z', color='map_group')
fig.show()

opt_clus(dr)
