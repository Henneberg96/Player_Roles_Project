import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, accuracy_score, f1_score, plot_roc_curve
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
import plotly.express as px
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from helpers.helperFunctions import *
from helpers.visualizations import *
from yellowbrick.classifier import ClassPredictionError, ROCAUC
import random
import statistics

#
def check_random_number_distribution(vals):
    sum = vals.value_counts().sum()
    counts = pd.DataFrame(vals.value_counts())
    counts['percentage'] = counts.iloc[:, 0].apply(lambda row: (row / sum) * 100)
    counts = counts.rename(columns={counts.columns[0]: "counts"})
    counts = counts.rename_axis('clusters').reset_index()
    plt.bar(counts['clusters'], counts['percentage'])
    plt.show()


data = event_data = pd.read_csv('C:/ITU/ITU_Research_Project/clustered_data/clusters_v2.csv',
                 sep=",",
                 encoding='unicode_escape')
df = data.drop(['playerId', 'seasonId', 'map_group', 'pos_group' ], axis = 1)

feature_columns = df.columns
input_variables = df.columns[feature_columns != 'ip_cluster']

input = df[input_variables]
target = df['ip_cluster']

cluster_and_weights = display_target_counts(df)
cluster_weights_dict = dict(zip(cluster_and_weights.ip_cluster, cluster_and_weights.percentage))


def produce_random_number(values, weights, length):
    predictions = random.choices(values, weights=(weights), k=length)
    return pd.DataFrame(predictions)


input_train, input_test, target_train, target_test = train_test_split(input, target, test_size=0.33, random_state=42)
random_weighted_predictions = produce_random_number(list(cluster_and_weights.ip_cluster), list(cluster_and_weights.percentage), input_test.shape[0])
mode_genereated_predictions = pd.DataFrame([statistics.mode(target_train)] * len(target_test))

check_random_number_distribution(random_weighted_predictions)

score = accuracy_score(target_test, random_weighted_predictions)

def evaluate_model(target_pred, target_test, modelType):
    f1 = f1_score(target_test, target_pred, average='weighted')
    cohen_kappa = cohen_kappa_score(target_test, target_pred)
    print('Model accuracy score ' + modelType + ': {0:0.4f}'. format(accuracy_score(target_test, target_pred)))
    print('F1 Score: ', "%.2f" % (f1*100))
    print('Cohen Kappa: ', "%.2f" % cohen_kappa)
    print(classification_report(target_test, target_pred))



def heatmap_probability_inspection(target_pred, target_test, title):
    matrix = confusion_matrix(target_test, target_pred)
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    fig = px.imshow(matrix, text_auto=True, aspect="auto")
    fig.update_layout( title=title)
    fig.show()

def display_results_classification_report(target_pred, target_test):
    report = classification_report(target_test, target_pred, output_dict=True)
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True)
    plt.show()


#Helper function
def show_feature_importances(model, input_train, input_test, target_test):

    # Genereate and print feature scores
    feature_scores = pd.Series(model.feature_importances_, index=input_train.columns).sort_values(ascending=False)
    fig = px.bar(feature_scores, orientation='h')
    fig.update_layout(
        title='Feature Importances',
        showlegend=False,
    )
    fig.show()

#Yellow brick libraries need model with train and test data to compute vizualisations

#Stacked bar chart showing prediction erros and how they are distributed on different classe
def vizualize_class_prediction_error(model, input_train, input_test, target_train, target_test):
    v = ClassPredictionError(model)
    v.fit(input_train, target_train)
    v.score(input_test, target_test)
    v.show()
    plt.show()

#Shows roc curve for each class and labels the class and teh auc score
def plot_ROC_curve(model, xtrain, ytrain, xtest, ytest):
        # Creating visualization with the readable labels
        visualizer = ROCAUC(model)

        # Fitting to the training data first then scoring with the test data
        visualizer.fit(xtrain, ytrain)
        visualizer.score(xtest, ytest)
        visualizer.show()


# Players to try and predict
#38021 De buyne
#3359 Messi
#217031 PArtey


grid_param = {
    'n_estimators' : [int(x) for x in np.linspace(start=50, stop=200, num=10)],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : range(2, 20, 1),
    'min_samples_split' : range(2, 10, 1),
    'min_samples_leaf' : range(1, 10, 1)
}

rf = RandomForestClassifier()
grid_search = RandomizedSearchCV(estimator = rf, param_distributions = grid_param, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
grid_search.fit(input_train, target_train)
print(grid_search.best_params_)

#Create models
rdf_model = RandomForestClassifier()
xgb_model = xgb.XGBClassifier()
ovr_model = OneVsRestClassifier(LinearSVC(random_state=42))
ovo_model = OneVsOneClassifier(LinearSVC(random_state=42))


#Fit models
ovr_model.fit(input_train, target_train)
ovo_model.fit(input_train, target_train)
rdf_model.fit(input_train, target_train)
xgb_model.fit(input_train, target_train)

ovr_preds = ovr_model.predict(input_test)
ovo_preds = ovo_model.predict(input_test)
rdf_preds = rdf_model.predict(input_test)
xgb_preds = xgb_model.predict(input_test)

evaluate_model(random_weighted_predictions, target_test, 'baseline')
evaluate_model(ovo_preds, target_test, 'ovo model')
evaluate_model(ovr_preds, target_test, 'ovr model')
evaluate_model(rdf_preds, target_test, 'rdf model')
evaluate_model(xgb_preds, target_test, 'xgb model')


heatmap_probability_inspection(random_weighted_predictions, target_test, 'Baseline Model')
heatmap_probability_inspection(ovo_preds, target_test, 'ovo model')
heatmap_probability_inspection(ovr_preds, target_test, 'ovr model')
heatmap_probability_inspection(rdf_preds, target_test, 'rdf model' )
heatmap_probability_inspection(rdf_preds, target_test, 'rdf model')

display_results_classification_report(ovo_preds, target_test)
display_results_classification_report(ovr_preds, target_test)
display_results_classification_report(rdf_preds, target_test)
display_results_classification_report(xgb_preds, target_test)

vizualize_class_prediction_error(rdf_model, input_train, input_test, target_train, target_test)
vizualize_class_prediction_error(xgb_model, input_train, input_test, target_train, target_test)
vizualize_class_prediction_error(ovo_model, input_train, input_test, target_train, target_test)
vizualize_class_prediction_error(ovr_model, input_train, input_test, target_train, target_test)

plot_ROC_curve(ovo_model, input_train, target_train, input_test, target_test )
plot_ROC_curve(ovr_model, input_train, target_train, input_test, target_test )
plot_ROC_curve(rdf_model, input_train, target_train, input_test, target_test )
plot_ROC_curve(xgb_model, input_train, target_train, input_test, target_test )

show_feature_importances(rdf_model, input_train, input_test, target_test)
show_feature_importances(xgb_model, input_train, input_test, target_test)

display_target_counts(data)

#Test of model on specific models
deBruyne = data[data['playerId'] == 38021]
messi = data[data['playerId'] == 3359]
partey = data[data['playerId'] == 217031]

deBruyne = deBruyne.drop(['playerId', 'seasonId', 'map_group', 'pos_group', 'ip_cluster' ], axis = 1)
messi = messi.drop(['playerId', 'seasonId', 'map_group', 'pos_group', 'ip_cluster' ], axis = 1)
partey = partey.drop(['playerId', 'seasonId', 'map_group', 'pos_group', 'ip_cluster' ], axis = 1)

de_buyne_proba = rdf_model.predict_proba(deBruyne)
messi_proba = rdf_model.predict_proba(messi)
partey_proba = rdf_model.predict_proba(partey)





'''

# Load dataset
DEF = pd.read_csv('C:/ITU/ITU_Research_Project/clustered_data/DEF.csv', sep=",", encoding='unicode_escape')
MID = pd.read_csv('C:/ITU/ITU_Research_Project/clustered_data/MID.csv', sep=",", encoding='unicode_escape')
WIDE = pd.read_csv('C:/ITU/ITU_Research_Project/clustered_data/WIDE.csv', sep=",", encoding='unicode_escape')
ATT = pd.read_csv('C:/ITU/ITU_Research_Project/clustered_data/ATT.csv', sep=",", encoding='unicode_escape')
raw = pd.read_csv("C:/ITU/ITU_Research_Project/preprocessed/events_CN.csv", sep = ",", encoding='unicode_escape')

#Merge clusters with features
DEF_v2 = pd.merge(DEF, raw, on=['playerId','seasonId', 'map_group', 'pos_group'])
MID_v2 = pd.merge(MID, raw, on=['playerId','seasonId', 'map_group', 'pos_group'])
WIDE_v2 = pd.merge(WIDE, raw, on=['playerId','seasonId', 'map_group', 'pos_group'])
ATT_v2 = pd.merge(ATT, raw, on=['playerId','seasonId', 'map_group', 'pos_group'])


#removal of redundant columns
DEF_v3 = DEF_v2.drop(['playerId', 'seasonId', 'map_group', 'pos_group', 'teamId'], axis=1)
MID_v3 = MID_v2.drop(['playerId', 'seasonId', 'map_group', 'pos_group', 'teamId'], axis=1)
ATT_v3 = ATT_v2.drop(['playerId', 'seasonId', 'map_group', 'pos_group', 'teamId'], axis=1)
WIDE_v3 = WIDE_v2.drop(['playerId', 'seasonId', 'map_group', 'pos_group', 'teamId'], axis=1)


# Extract input and target values for each pos_gropu
feature_columns = DEF_v3.columns
input_variables = DEF_v3.columns[feature_columns != 'cluster']

# Def
def_input = DEF_v3[input_variables]
def_target = DEF_v3['cluster']

# mid
mid_input = MID_v3[input_variables]
mid_target = MID_v3['cluster']

# wing
wide_input = WIDE_v3[input_variables]
wide_target = WIDE_v3['cluster']

# att
att_input = ATT_v3[input_variables]
att_target = ATT_v3['cluster']

# Split of datasets for each position
def_input_train, def_input_test, def_target_train, def_target_test = train_test_split(def_input, def_target,
                                                                     test_size=0.33, random_state=42)

mid_input_train, mid_input_test, mid_target_train, mid_target_test = train_test_split(mid_input, mid_target,
                                                                     test_size=0.33, random_state=42)

wide_input_train, wide_input_test, wide_target_train, wide_target_test = train_test_split(wide_input, wide_target,
                                                                         test_size=0.33, random_state=42)

att_input_train, att_input_test, att_target_train, att_target_test = train_test_split(att_input, att_target,
        
        #MID
perform_regression(model, mid_input_train, mid_input_test, mid_target_train, mid_target_test, "MID")
#WIDE
perform_regression(model, wide_input_train, wide_input_test, wide_target_train, wide_target_test, "WIDE")
#ATT
perform_regression(model, att_input_train, att_input_test, att_target_train, att_target_test, "ATT")
                                                             test_size=0.33, random_state=42)

'''