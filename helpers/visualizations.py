import numpy as np
import matplotlib.pyplot as plt
from mplsoccer import Pitch

#Define pitch with predeefiend rows
def createPitchWithZones():
    data = np.array([
        [0, 0],
        [16,0], # x+16
        [0, 19],
        [16, 19], # x+17
        [33, 0],
        [33, 19],
        [67, 0], # x+34
        [67, 19],
        [84, 0],
        [84, 19],
        [100, 0],
        [100, 19],
        [0, 81],
        [0, 100],
        [16, 81],
        [16, 100],
        [33, 81],
        [33, 100],
        [67, 81],  # x+34
        [67, 100],
        [84, 81],
        [84, 100],
        [100, 81],
        [100, 100],
        [16, 63],
        [16, 63],
        [16, 63],
        [33, 63],
        [67, 63],
        [16, 37],
        [33, 37],
        [50, 19],
        [50, 0],
        [50, 37],
        [50, 63],
        [50, 81],
        [50, 100],
        [67, 37],
        [84, 37],
        [84, 63],])
    pitch = Pitch(pitch_type='wyscout', axis=True,
                  positional=True,
                  tick=True,
                  label=True)
    fig, ax = pitch.draw()
    x, y = data.T
    plt.scatter(x,y)
    plt.show()

#Draw customized polygons
#shape1 = np.array([[0, 0], [50, 0], [50, 50], [0, 50]])
#shape2 = np.array([[0, 50], [50, 50], [50, 100], [0, 100]])
#shape3 = np.array([[50, 0], [100, 0], [100, 50], [50, 50]])
#shape4 = np.array([[50, 50], [100, 50], [100, 100], [50, 100]])
#areas = [shape1, shape2, shape3, shape4]
#pitch.polygon(areas , edgecolor='black', fc=[1, 0, 0], alpha=0.3, ax=ax)
