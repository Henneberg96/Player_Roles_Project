import pandas as pd

def findArea(row):
    s = ""
  #  id = row['id']
    sub_e = row['subEventName']
    x = row['x']
    y = row['y']
    if(x >= 0 and x <= 50 and y >=0 and y <=50):
        s = '_01'
    elif(x > 50 and x <100 and y >0 and  y <50):
        s = '_02'
    elif(x > 0 and x <50 and y >50 and  y <100):
        s = '_03'
    elif(x > 50 and x <100 and y >50 and y <100):
        s = '_04'
    else: s = "out_of_bounce"
    return sub_e + s

