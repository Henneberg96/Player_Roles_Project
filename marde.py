import math
import numpy as np
import pandas as pd
from helpers.helperFunctions import *

# load event file
df = pd.read_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/Wyscout_Events - Copy.csv',
                 sep=";",
                 encoding='unicode_escape')

# drop first columns
df.drop(columns=df.columns[0], axis=1, inplace=True)

df.shape  # 73 columns at this point

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

df.shape  # 33 columns at this point

# drop irelevant sub events
df = df[df.subEventName.notnull()]
df = df[df.subEventName != 'Whistle']

# merge with matches dataset to get season ID
matches = pd.read_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/Wyscout_Matches.csv',
                      sep=";",
                      encoding='unicode_escape')
df = pd.merge(df, matches, on='matchId')
compInd = df.pop('competitionId')
df.insert(5, 'competitionId', compInd, allow_duplicates=True)
sznInd = df.pop('seasonId')
df.insert(6, 'seasonId', sznInd, allow_duplicates=True)

df.shape  # 35 columns at this point

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
df.shape  # 57 columns at this point

# defining event zones
df.insert(9, 'eventZone', df.apply(lambda row: findArea(row), axis=1), allow_duplicates=True)
dfx = df.copy()
dfy = zones(dfx, temp)
dfy.shape

# adding shot, passing, and crossing distances
df['shot_distance'] = np.where(df['subEventName'] == 'Shot', ec(df['x'], df['end_x'], df['y'], df['end_y']), np.nan)
df['passing_distance'] = np.where(df['eventName'] == 'Pass', ec(df['x'], df['end_x'], df['y'], df['end_y']), np.nan)
df['cross_distance'] = np.where(df['subEventName'] == 'Cross', ec(df['x'], df['end_x'], df['y'], df['end_y']), np.nan)

#adding progressive passes
df['progressive_passes'] = np.where(df['eventName'] == 'Pass', pp(df.x, df.end_x), np.nan)
df['progressive_passes_distance'] = np.where(df['progressive_passes'] == 1.00000, ec(df['x'], df['end_x'], df['y'], df['end_y']), np.nan)

#adding extra passes types
df['pass_direction'] = np.where(df['eventName'] == 'Pass', direction(df.x, df.end_x), np.nan)

#adding swtiches
df['pass_direction'] = np.where(df['eventName'] == 'Pass', switch(df.y, df.end_y), np.nan)

df.shape #64 columns at this point

#creating dummies for sub events and merging
dfx = pd.get_dummies(df['subEventName'])
dfx[temp] = dfx[temp].replace({'0':np.nan, 0:np.nan})
dfx['eventNameId'] = df['eventNameId']
df = pd.merge(df, dfx, on='eventNameId')

df.shape #99 columns at this point