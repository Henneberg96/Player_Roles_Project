import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn import metrics


# Filter to determine where an event occured
def findArea(row):
    s = ""
    #  id = row['id']
    # print(row)
    x = row['x']
    y = row['y']
    if (x >= 0 and x <= 16 and y >= 0 and y <= 19):
        s = 1
    elif (x > 16 and x <= 33 and y >= 0 and y <= 19):
        s = 5
    elif (x > 33 and x <= 50 and y >= 0 and y <= 19):
        s = 9
    elif (x > 50 and x <= 67 and y >= 0 and y <= 19):
        s = 15
    elif (x > 67 and x <= 84 and y >= 0 and y <= 19):
        s = 19
    elif (x > 84 and x <= 100 and y >= 0 and y <= 19):
        s = 24
    elif (x > 16 and x <= 33 and y > 19 and y <= 37):
        s = 7
    elif (x > 33 and x <= 50 and y > 19 and y <= 37):
        s = 11
    elif (x >= 50 and x <= 67 and y > 19 and y <= 37):
        s = 17
    elif (x > 67 and x <= 84 and y > 19 and y <= 37):
        s = 21
    elif (x > 16 and x <= 33 and y > 37 and y <= 63):
        s = 8
    elif (x > 33 and x <= 50 and y > 37 and y <= 63):
        s = 13
    elif (x > 50 and x <= 67 and y > 37 and y <= 63):
        s = 18
    elif (x > 67 and x <= 84 and y > 37 and y <= 63):
        s = 23
    elif (x > 16 and x <= 33 and y > 63 and y <= 81):
        s = 6
    elif (x > 33 and x <= 50 and y > 63 and y <= 81):
        s = 12
    elif (x > 50 and x <= 67 and y > 63 and y <= 81):
        s = 16
    elif (x > 67 and x <= 84 and y > 63 and y <= 81):
        s = 22
    elif (x >= 0 and x <= 16 and y > 81 and y <= 100):
        s = 2
    elif (x > 16 and x <= 33 and y > 81 and y <= 100):
        s = 4
    elif (x > 33 and x <= 50 and y > 81 and y <= 100):
        s = 10
    elif (x > 50 and x <= 67 and y > 81 and y <= 100):
        s = 14
    elif (x > 67 and x <= 84 and y > 81 and y <= 100):
        s = 20
    elif (x > 84 and x <= 100 and y > 81 and y <= 100):
        s = 25
    elif (x >= 0 and x <= 16 and y > 19 and y <= 81):
        s = 3
    elif (x >= 84 and x <= 100 and y > 19 and y <= 81):
        s = 26
    else:
        s = 0
    return s

def ec(x1, x2, y1, y2):
    return np.sqrt(pow((x2 - x1), 2) + pow((y2 - y1), 2))


def pp(x1, x2):
    dist = x2 - x1
    return np.where(x1 > 50, np.where(dist > 10, 1.00000, np.nan),
                    np.where(dist > 25, 1.00000, np.nan))


def direction(x1, x2):
    dist = x2 - x1
    return np.where(dist > 4, 'forward',
                    np.where(dist < -4, 'backward', 'horizontal'))


def switch(y1, y2):
    dist = y2 - y1
    return np.where(dist > 35, 1.00000, np.nan)


def gmode(df):
    temp = df.groupby(['playerId', 'seasonId']).obj.columns
    temp = temp.drop('playerId')
    temp = temp.drop('seasonId')
    print(temp)
    gtemp1 = df.groupby(['playerId', 'seasonId'], as_index=False)['Simple pass_zone'].agg(pd.Series.mode)
    gtemp1.pop('Simple pass_zone')
    for i in temp:
        print(i)
        gtemp2 = df.groupby(['playerId', 'seasonId'], as_index=False)[i].agg(gmodeHelp)
        print(gtemp2)
        gtemp1 = pd.merge(gtemp1, gtemp2, on=['playerId', 'seasonId'])
        print(gtemp1)
    return gtemp1


def gmodeHelp(x):
    m = pd.Series.mode(x)
    return m.values[0] if not m.empty else np.nan

def pos_group(row):
    x = row['map_group']
    g = ['GK']
    d = ['CB']
    m = ['AM', 'CM', 'DM']
    w = ['LM', 'RM', 'LW', 'RW', 'RB', 'LB', 'LWB', 'RWB']
    f = ['FW']
    if x in g:
        return "GK"
    elif x in d:
        return "DEF"
    elif x in m:
        return "MID"
    elif x in w:
        return "WIDE"
    elif x in f:
        return "ATT"
    else:
        return "other"


def off_def(row):
    x = row['map_group']
    off = ['FW', 'LM', 'RM', 'LW', 'RW', 'AM', 'CM']
    deff = ['CB', 'RB', 'LB', 'LWB', 'RWB', 'DM']
    if x in off:
        return "off"
    elif x in deff:
        return "def"
    else:
        return "other"

def opt_clus(dr):
    n_range = range(2, 21)
    bic_score = []
    aic_score = []

    for n in n_range:
        gm = GaussianMixture(n_components=n, covariance_type='full', random_state=42)
        gm.fit(dr)
        labels = gm.predict(dr)
        #labels = labels.reshape(labels.shape[0], 1) # for 3D
        print(("Clusters: ", n, "Siloutte: ", metrics.silhouette_score(dr, labels, metric='euclidean')))
        bic_score.append(gm.bic(dr))
        aic_score.append(gm.aic(dr))

    fig, ax = plt.subplots(figsize=(12, 8), nrows=1)
    ax.plot(n_range, bic_score, '-o', color='orange', label='BIC')
    ax.plot(n_range, aic_score, '-o', color='blue', label='AIC')
    ax.set(xlabel='Number of Clusters', ylabel='Score')
    ax.set_xticks(n_range)
    ax.set_title('BIC and AIC Scores Per Number Of Clusters')
    ax.legend(fontsize='x-large')
    plt.show()


def gmm_to_df(df, phase):
    if phase == 'ip':
        frame = pd.DataFrame(df.reshape(df.shape[0], 1), columns=["ip_cluster"])
    elif phase == 'op':
        frame = pd.DataFrame(df.reshape(df.shape[0], 1), columns=["op_cluster"])
    return frame


# Identifying highly correlated features
def find_outliers_IQR(df):
   q1=df.quantile(0.25)
   q3=df.quantile(0.75)
   IQR=q3-q1
   outliers = df[((df<(q1-1.5*IQR)) | (df>(q3+1.5*IQR)))]
   return outliers


def names_clusters(data, cluster):
    df = data.iloc[:, np.r_[0, 5, 6:12]]
    players = pd.read_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/Wyscout_Players.csv', sep=";", encoding='unicode_escape')
    players.drop(columns=players.columns[0], axis=1, inplace=True)
    dfp = pd.merge(players, df, on='playerId')
    dfp = dfp[dfp.ip_cluster == cluster]
    dfp = dfp.iloc[:, np.r_[4, 2, 1, 18, 17, 15, 16, 13, 14]]
    return dfp
