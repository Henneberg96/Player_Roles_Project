import numpy as np
import pandas as pd
from helpers.helperFunctions import findArea
from helpers.visualizations import createPitchWithZones
from helpers.helperFunctions import loadData

import matplotlib.pyplot as plt
from mplsoccer import Pitch

#Load data and utilize methods to draw pitch with zones and attach zones to events
event_data = pd.read_csv ('C:/ITU/ITU_Research_Project/WyScout_Data_Two/NewEvents/Wyscout_Events.csv', sep=";")
event_data = event_data[event_data.subEventName.notnull()]
event_data['event_zone'] = event_data.apply(lambda row: findArea(row), axis=1)
event_data.head(5)
createPitchWithZones()