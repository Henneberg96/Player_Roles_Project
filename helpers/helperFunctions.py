import pandas as pd
import numpy as np


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
        s = 2
    elif (x > 33 and x <= 50 and y >= 0 and y <= 19):
        s = 3
    elif (x > 50 and x <= 67 and y >= 0 and y <= 19):
        s = 4
    elif (x > 67 and x <= 84 and y >= 0 and y <= 19):
        s = 5
    elif (x > 84 and x <= 100 and y >= 0 and y <= 19):
        s = 6
    elif (x > 16 and x <= 33 and y > 19 and y <= 37):
        s = 7
    elif (x > 33 and x <= 50 and y > 19 and y <= 37):
        s = 8
    elif (x >= 50 and x <= 67 and y > 19 and y <= 37):
        s = 9
    elif (x > 67 and x <= 84 and y > 19 and y <= 37):
        s = 10
    elif (x > 16 and x <= 33 and y > 37 and y <= 63):
        s = 12
    elif (x > 33 and x <= 50 and y > 37 and y <= 63):
        s = 13
    elif (x > 50 and x <= 67 and y > 37 and y <= 63):
        s = 14
    elif (x > 67 and x <= 84 and y > 37 and y <= 63):
        s = 15
    elif (x > 16 and x <= 33 and y > 63 and y <= 81):
        s = 17
    elif (x > 33 and x <= 50 and y > 63 and y <= 81):
        s = 18
    elif (x > 50 and x <= 67 and y > 63 and y <= 81):
        s = 19
    elif (x > 67 and x <= 84 and y > 63 and y <= 81):
        s = 20
    elif (x >= 0 and x <= 16 and y > 81 and y <= 100):
        s = 21
    elif (x > 16 and x <= 33 and y > 81 and y <= 100):
        s = 22
    elif (x > 33 and x <= 50 and y > 81 and y <= 100):
        s = 23
    elif (x > 50 and x <= 67 and y > 81 and y <= 100):
        s = 24
    elif (x > 67 and x <= 84 and y > 81 and y <= 100):
        s = 25
    elif (x > 84 and x <= 100 and y > 81 and y <= 100):
        s = 26
    elif (x >= 0 and x <= 16 and y > 19 and y <= 81):
        s = 11
    elif (x >= 84 and x <= 100 and y > 19 and y <= 81):
        s = 16
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
