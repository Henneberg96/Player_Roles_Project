import pandas as pd
import numpy as np

#Filter to determine where an event occured
def findArea(x, y):
    s = ""
  #  id = row['id']
    #print(row)
    #x = row['x']
    #y = row['y']
    if(x >= 0 and x <= 16 and y >=0 and y <=19):
        s = 1
    elif(x > 16 and x <=33 and y >=0 and  y <=19):
        s = 2
    elif (x > 33 and x <= 50 and y >= 0 and y <= 19):
        s = 3
    elif (x > 50 and x <= 67 and y >= 0 and y <= 19):
        s = 4
    elif (x > 67 and x <= 84 and y >= 0 and y <= 19):
        s = 5
    elif (x > 84  and x <=100 and y >= 0 and y <= 19):
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
    else: s = 0
    return s

def findArea2(x, y):
    s = ""
  #  id = row['id']
    #print(row)
    #x = row['x']
    #y = row['y']
    if(x >= 0 and x <= 16 and y >=0 and y <=19):
        s = 1
    else:
        s = 0
    return s


def zones(df, temp):
    df1 = df.iloc[:1000000, :]
    #df2 = df.iloc[1000000:2000000, :]
    #df3 = df.iloc[2000000:3000000, :]
    #df4 = df.iloc[3000000:4000000, :]
    #df5 = df.iloc[4000000:, :]
    for i in temp:
        name = i + '_zone'
        df1[name] = np.where(df1['subEventName'] == i, findArea2(df1['x'], df1['y']), np.nan)
    return df1
    for i in temp:
        name = i + '_zone'
        df2[name] = np.where(df2['subEventName'] == i, df2.apply(lambda row: findArea(row)), np.nan)
    for i in temp:
        name = i + '_zone'
        df3[name] = np.where(df3['subEventName'] == i, df3.apply(lambda row: findArea(row)), np.nan)
    for i in temp:
        name = i + '_zone'
        df4[name] = np.where(df4['subEventName'] == i, df4.apply(lambda row: findArea(row)), np.nan)
    for i in temp:
        name = i + '_zone'
        df5[name] = np.where(df5['subEventName'] == i, df5.apply(lambda row: findArea(row)), np.nan)

    print(df1.shape, df2.shape, df3.shape, df4.shape, df5.shape)


def ec(x1, x2, y1, y2):
    return np.sqrt(pow((x2 - x1), 2) + pow((y2 - y1), 2))


def pp(x1, x2):
    dist = x2 - x1
    return np.where(x1 > 50, np.where(dist > 10, 1.00000, np.nan),
             np.where(dist > 25, 1.00000, np.nan))

def direction(x1, x2):
    dist = x2 - x1
    return np.where(dist > 2, 'forward',
             np.where(dist < -2, 'backward', 'horizontal'))

def switch(y1, y2):
    dist = y2 - y1
    return np.where(dist > 35, 1.00000, np.nan)