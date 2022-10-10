import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# Load dataset
df = pd.read_csv('C:/ITU/ITU_Research_Project/preprocessed/events_CN.csv', sep=",", encoding='unicode_escape')

df = df.drop(['playerId', 'seasonId', 'teamId', 'map_group'], axis=1)

# Provide random clusters - only temporary
df['clusters'] = np.random.randint(1, 5, size=(len(df), 1))

# Positions
print(df["pos_group"].unique())

# Extracting dataframes based on position group
df_def = df.loc[df['pos_group'] == "DEF"]
df_mid = df.loc[df['pos_group'] == "MID"]
df_wing = df.loc[df['pos_group'] == "WING"]
df_att = df.loc[df['pos_group'] == "ATT"]

# Extract input and target values for each pos_gropu
feature_columns = df.columns
input_variables = df.columns[feature_columns != 'clusters']

# Def
def_input = df_def[input_variables]
def_target = df_def['clusters']
def_input.pop(def_input.columns[-1])

# mid
mid_input = df_mid[input_variables]
mid_target = df_mid['clusters']
mid_input.pop(mid_input.columns[-1])

# wing
wing_input = df_wing[input_variables]
wing_target = df_wing['clusters']
wing_input.pop(wing_input.columns[-1])

# att
att_input = df_att[input_variables]
att_target = df_att['clusters']
att_input.pop(att_input.columns[-1])

# Split of datasets for each position
def_input_train, def_test, def_target_train, target_test = train_test_split(def_input, def_target,
                                                                            test_size=0.33, random_state=42)
#Check shape
def_input_train.shape
def_test.shape

# Creation of randomForestClassifier model
rnd_clf_def = RandomForestClassifier(
    n_estimators=100,
    criterion='gini',
    max_depth=4)

# Fit model for each position
rnd_clf_def.fit(def_input_train, def_target_train)
# Predict
def_target_pred = rnd_clf_def.predict(def_test)

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(target_test, def_target_pred)))

#Genereate and print feature scores
feature_scores = pd.Series(rnd_clf_def.feature_importances_, index=def_input_train.columns).sort_values(ascending=False)

# Create and display an vizualization of feature importances
sns.barplot(x=feature_scores, y=feature_scores.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Features importances")
plt.show()

print(classification_report(target_test, def_target_pred))