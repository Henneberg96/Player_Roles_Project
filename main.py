import numpy as np
import pandas as pd
from helpers.helperFunctions import findArea

import matplotlib.pyplot as plt
from mplsoccer import Pitch

event_data = pd.read_csv ('C:/ITU/ITU_Research_Project/WyScout_Data_Two/NewEvents/Wyscout_Events.csv', sep=";")
subEvents = event_data.loc[:, ['id', 'subEventName', 'x', 'y']]

subEvents_with_Zones = df = pd.DataFrame(columns = ['id', 'event_zone'])
event_data = event_data[event_data.subEventName.notnull()]

event_data['event_zone'] = event_data.apply(lambda row: findArea(row), axis=1)



pitch = Pitch(pitch_type='wyscout', axis=True, label=True, tick=True)
fig, ax = pitch.draw()
shape1 = np.array([[0, 0], [50, 0], [50, 50], [0, 50]])
shape2 = np.array([[0, 50], [50, 50], [50, 100], [0, 100]])
shape3 = np.array([[50, 0], [100, 0], [100, 50], [50, 50]])
shape4 = np.array([[50, 50], [100, 50], [100, 100], [50, 100]])
areas = [shape1, shape2, shape3, shape4]
pitch.polygon(areas , edgecolor='black', fc=[1, 0, 0], alpha=0.3, ax=ax)
plt.show()



