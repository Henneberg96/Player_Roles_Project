import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from helpers.helperFunctions import *

# load event file
df = pd.read_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/Wyscout_Events - Copy.csv',
                 sep=";",
                 encoding='unicode_escape')

# drop first columns
df.drop(columns=df.columns[0], axis=1, inplace=True)

print(df.shape)  # 73 columns at this point

# drop columns that won't be used
df = df.drop(['left_foot',
              'right_foot',
              'take_on_left',
              'take_on_right',
              'free_space_left',
              'free_space_right',
              'high_cross',
              'low_cross',
              'lost',
              'won',
              'neutral',
              'clearance',
              'fairplay',
              'direct',
              'indirect',
              'low',
              'low_right',
              'center',
              'left',
              'low_left',
              'right',
              'high',
              'high_left',
              'high_right',
              'miss_low_right',
              'miss_left',
              'miss_low_left',
              'miss_right',
              'miss_high',
              'miss_high_left',
              'miss_high_right',
              'post_low_right',
              'post_left',
              'post_low_left',
              'post_right',
              'post_high',
              'post_high_left',
              'post_high_right',
              'blocked',
              'eventId'],
             axis=1)

print(df.shape)  # 33 columns at this point

# drop irelevant sub events
df = df[df.subEventName.notnull()]
df = df[df.subEventName != 'Whistle']
df = df[df.playerId != 0]

# merge with matches dataset to get season ID
matches = pd.read_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/Wyscout_Matches.csv',
                      sep=";",
                      encoding='unicode_escape')
df = pd.merge(df, matches, on='matchId')
compInd = df.pop('competitionId')
df.insert(5, 'competitionId', compInd, allow_duplicates=True)
sznInd = df.pop('seasonId')
df.insert(6, 'seasonId', sznInd, allow_duplicates=True)

print(df.shape)  # 35 columns at this point

# rearranging columns
eIdInd = df.pop('id')
df.insert(6, 'eventNameId', eIdInd, allow_duplicates=True)
eNameInd = df.pop('eventName')
df.insert(6, 'eventName', eNameInd, allow_duplicates=True)
subEventInd = df.pop('subEventId')
df.insert(7, 'subEventId', subEventInd, allow_duplicates=True)
subEventNameInd = df.pop('subEventName')
df.insert(8, 'subEventName', subEventNameInd, allow_duplicates=True)

# append df with accuracy per sub event
temp = pd.unique(df['subEventName'])
for i in temp:
    name = i + '_acc'
    df[name] = np.where(df['subEventName'] == i, df['accurate'], np.nan)
for i in temp:
    name = i + '_acc'
    if df[name].isnull().all():
        df.drop(name, axis=1, inplace=True)

df_p = df[['subEventName', 'accurate', 'Cross_acc']]  # test
print(df.shape)  # 57 columns at this point

# defining event zones
df.insert(9, 'eventZone', df.apply(lambda row: findArea(row), axis=1), allow_duplicates=True)
for i in temp:
    name = i + '_zone'
    df[name] = np.where(df['subEventName'] == i, df['eventZone'], np.nan)

# adding shot, passing, and crossing distances
df['shot_distance'] = np.where(df['subEventName'] == 'Shot', ec(df['x'], df['end_x'], df['y'], df['end_y']), np.nan)
df['passing_distance'] = np.where(df['eventName'] == 'Pass', ec(df['x'], df['end_x'], df['y'], df['end_y']), np.nan)
df['cross_distance'] = np.where(df['subEventName'] == 'Cross', ec(df['x'], df['end_x'], df['y'], df['end_y']), np.nan)

# adding progressive passes
df['progressive_passes'] = np.where(df['eventName'] == 'Pass', pp(df.x, df.end_x), np.nan)
df['progressive_passes_distance'] = np.where(df['progressive_passes'] == 1.00000,
                                             ec(df['x'], df['end_x'], df['y'], df['end_y']), np.nan)

# adding extra passes types
df['pass_direction'] = np.where(df['eventName'] == 'Pass', direction(df.x, df.end_x), np.nan)

# adding swtiches
df['switches'] = np.where(df['eventName'] == 'Pass', switch(df.y, df.end_y), np.nan)

print(df.shape)  # 100 columns at this point

# creating dummies for pass direction and sub events and merging
temp2 = pd.unique(df['pass_direction'])
dfx = pd.get_dummies(df['pass_direction'])
dfx[temp2] = dfx[temp2].replace({'0': np.nan, 0: np.nan})
dfx['eventNameId'] = df['eventNameId']
dfx.pop('nan')
df = pd.merge(df, dfx, on='eventNameId')

print(df['backward'].value_counts(), '\n \n', df['horizontal'].value_counts(), '\n \n',
      df['forward'].value_counts())  # test

dfx = pd.get_dummies(df['subEventName'])
dfx[temp] = dfx[temp].replace({'0': np.nan, 0: np.nan})
dfx['eventNameId'] = df['eventNameId']
df = pd.merge(df, dfx, on='eventNameId')

print(df.shape)  # 138 columns at this point

# saving df raw
df_raw = df.copy()  # revert by df=df_raw.copy()

# reshaping to player per season stats
df_sum = df.iloc[:, np.r_[0, 4, 16:58, 96, 99:138]]
df_freq = df.iloc[:, np.r_[0, 1, 4, 58:93]]
df_avg = df.iloc[:, np.r_[0, 4, 93:96, 97]]
df_other = df.iloc[:, np.r_[0, 2:16]]

df_sum = df_sum.groupby(['playerId', 'seasonId'], as_index=False).sum()
df_freq = gmode(df_freq)
df_avg = df_avg.groupby(['playerId', 'seasonId'], as_index=False).mean()

dfc = pd.merge(df_sum, df_freq, on=['playerId', 'seasonId'])
dfc = pd.merge(dfc, df_avg, on=['playerId', 'seasonId'])

# creating accuracy percentages
temp3 = dfc.filter(regex='_acc$', axis=1)
temp3 = temp3.columns
temp3 = [x[:-4] for x in temp3]
for i in temp3:
    acc = i + '_acc'
    name = i + '_acc_percentage'
    dfc[name] = dfc[acc] / dfc[i] * 100

print(dfc.shape)  # 146 columns at this point

# exporting dataframe
dfc.to_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/events_cleaned.csv', index=False)

# importing cleaned df
dfc = pd.read_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/events_cleaned.csv',
                 sep=",",
                 encoding='unicode_escape')

# stat ratios
dfc['goals_pr_shot'] = dfc['goal'] / dfc['Shot']
dfc['pp_pass_ratio'] = dfc['progressive_passes'] / (dfc['Cross'] + dfc['Hand pass'] + dfc['Head pass'] + dfc['High pass'] + dfc['Launch'] + dfc['Simple pass'] + dfc['Smart pass'])
dfc['key_pass_ratio'] = dfc['key_pass'] / (dfc['Cross'] + dfc['Hand pass'] + dfc['Head pass'] + dfc['High pass'] + dfc['Launch'] + dfc['Simple pass'] + dfc['Smart pass'])
dfc['acc_ratio'] = dfc['accurate'] / dfc['not_accurate']
dfc['counter_opportunity_ratio'] = dfc['counter_attack'] / dfc['opportunity']
dfc['anticipation_percentage'] = dfc['anticipated'] / dfc['Ground defending duel']
dfc['interception_percentage'] = dfc['interception'] / (dfc['Ground loose ball duel'] + dfc['Ground attacking duel'])
dfc['slide_tackle_ratio'] = dfc['sliding_tackle'] / (dfc['Ground loose ball duel'] + dfc['Ground attacking duel'] + dfc['Ground defending duel'])
dfc['dangerous_duel_loss_ratio'] = dfc['dangerous_ball_lost'] / (dfc['Ground loose ball duel'] + dfc['Ground attacking duel'])

# merging with played minutes
minutes = pd.read_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/Wyscout_Positions_Minutes.csv',
                 sep=";",
                 encoding='unicode_escape')
minutes.pop('Unnamed: 0')
minutes.pop('teamId')
minutes.pop('matchId')
minutes.pop('position')
minutes = minutes.groupby(['playerId', 'seasonId'], as_index=False).sum()
dfc = pd.merge(dfc, minutes, on=['playerId', 'seasonId'])

# normalizing with per 90
dfc = dfc[dfc.time > 500] # cutoff minutes
dfc['games'] = dfc['time'] / 90
df_id = dfc.iloc[:, np.r_[0, 1]]
df_norm = dfc.iloc[:, np.r_[0, 1, 2:83, 156]]
df_none = dfc.iloc[:, np.r_[0, 1, 84:154]]

df_norm = df_norm.iloc[:,2:83].div(df_norm.games, axis=0)

dfn = pd.merge(df_id, df_norm, left_index=True, right_index=True)
dfn = pd.merge(dfn, df_none, on=['playerId', 'seasonId'])

# moving teamId to the front
tId = dfn.pop('teamId')
dfn.insert(2, 'teamId', tId, allow_duplicates=True)

# further normalization - scaling
scale = StandardScaler()
dfn_id = dfn.iloc[:, np.r_[0:3]]
dfn_scale = dfn.iloc[:, np.r_[3:153]]

dfn_scale = dfn_scale[dfn_scale.columns] = scale.fit_transform(dfn_scale[dfn_scale.columns])

dfn['slide_tackle_ratio'].dtypes