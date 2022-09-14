import pandas as pd

def findArea(row):
    s = ""
  #  id = row['id']
    sub_e = row['subEventName']
    x = row['x']
    y = row['y']
    if(x >= 0 and x <= 16 and y >=0 and y <=19):
        s = '_01'
    elif(x >= 17 and x <=33 and y >=0 and  y <=19):
        s = '_02'
    elif (x >= 34 and x <= 50 and y >= 0 and y <= 19):
        s = '_03'
    elif (x >= 51 and x <= 67 and y >= 0 and y <= 19):
        s = '_04'
    elif (x >= 68 and x <= 84 and y >= 0 and y <= 19):
        s = '_05'
    elif (x >= 85  and x <=100 and y >= 0 and y <= 19):
        s = '_06'
    elif (x >= 17 and x <= 33 and y >= 20 and y <= 37):
        s = '_07'
    elif (x >= 34 and x <= 50 and y >= 20 and y <= 37):
        s = '_08'
    elif (x >= 51 and x <= 67 and y >= 20 and y <= 37):
        s = '_09'
    elif (x >= 68 and x <= 84 and y >= 20 and y <= 37):
        s = '_10'
    elif (x >= 17 and x <= 33 and y >= 38 and y <= 63):
        s = '_12'
    elif (x >= 34 and x <= 50 and y >= 38 and y <= 63):
        s = '_13'
    elif (x >= 51 and x <= 67 and y >= 38 and y <= 63):
        s = '_14'
    elif (x >= 68 and x <= 84 and y >= 38 and y <= 63):
        s = '_15'
    elif (x >= 17 and x <= 33 and y >= 64 and y <= 81):
        s = '_17'
    elif (x >= 34 and x <= 50 and y >= 64 and y <= 81):
        s = '_18'
    elif (x >= 51 and x <= 67 and y >= 64 and y <= 81):
        s = '_19'
    elif (x >= 68 and x <= 84 and y >= 64 and y <= 81):
        s = '_20'
    elif (x >= 0 and x <= 16 and y >= 81 and y <= 100):
            s = '_21'
    elif (x >= 17 and x <= 33 and y >= 81 and y <= 100):
        s = '_22'
    elif (x >= 34 and x <= 50 and y >= 81 and y <= 100):
        s = '_23'
    elif (x >= 51 and x <= 67 and y >= 81 and y <= 100):
        s = '_24'
    elif (x >= 68 and x <= 84 and y >= 81 and y <= 100):
        s = '_25'
    elif (x >= 85 and x <= 100 and y >= 81 and y <= 100):
        s = '_26'
    elif (x >= 0 and x <= 16 and y >= 20 and y <= 80):
        s = '_11'
    elif (x >= 84 and x <= 100 and y >= 20 and y <= 80):
        s = '_16'
    else: s = "out_of_bounce"
    return sub_e + s

