import numpy as np
import pandas as pd

#load event file
df = pd.read_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/Wyscout_Events - Copy.csv',
                   sep=";",
                   encoding='unicode_escape')

#drop first columns
df.drop(columns=df.columns[0], axis=1, inplace=True)

df.shape #73 columns at this point

#drop columns that won't be used
df = df.drop(['left_foot',
                  'right_foot',
                  'head',
                  'take_on_left',
                  'take_on_right',
                  'free_space_left',
                  'free_space_right',
                  'high_cross',
                  'low_cross',
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
                  'blocked'],
                 axis=1)

df.shape #36 columns at this point

#drop irelevant sub events
df = df[df.subEventName.notnull()]
df = df[df.subEventName != 'Whistle']

#merge with matches dataset to get season ID
matches = pd.read_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/Wyscout_Matches.csv',
                   sep=";",
                   encoding='unicode_escape')
df = pd.merge(df, matches, on='matchId')
compInd = df.pop('competitionId')
df.insert(5, 'competitionId', compInd, allow_duplicates=True)
sznInd = df.pop('seasonId')
df.insert(6, 'seasonId', sznInd, allow_duplicates=True)

df.shape #38 columns at this point

#rearranging columns
eIdInd = df.pop('id')
df.insert(6, 'eventNameId', eIdInd, allow_duplicates=True)
eNameInd = df.pop('eventName')
df.insert(6, 'eventName', eNameInd, allow_duplicates=True)
df.pop('eventTypeId')
subEventInd = df.pop('subEventId')
df.insert(7, 'subEventId', subEventInd, allow_duplicates=True)
subEventNameInd = df.pop('subEventName')
df.insert(8, 'subEventName', subEventNameInd, allow_duplicates=True)

df.shape #37 columns at this point

#creating df copy for backup
dfx = df.copy()

#append df with accuracy per sub event
temp = pd.unique(df['subEventName'])
for i in temp:
    name = i + '_acc'
    dfx[name] = np.where(dfx['subEventName'] == i, dfx['accurate'], np.nan)

df_p = dfx[['subEventName', 'accurate', 'Cross_acc']] #test
dfx.shape #72 columns at this point