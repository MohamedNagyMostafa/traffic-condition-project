import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns


data = pd.read_csv('../output/The M25 anticlockwise between junctions J13  and J12  .csv')


dict = defaultdict(list)
for i, date_selection in enumerate(pd.to_datetime(data['date'])):
    if date_selection.day == 28:
        dict[data['sensor'].values[i]].append(date_selection.hour + (date_selection.minute/60))

colors = ['r', 'g', 'b']

sns.set()
fig, ax = plt.subplots()
fig.suptitle('Overlapped Readings of Multiple Sensors for anticlockwise \nbetween junctions J13  and J12 on 28 Jan.2017', fontsize=20)

y_label = []
l1 = np.array((8, 8))
for i, sensor in enumerate(dict.keys()):
    ax.plot(dict[sensor], [i] * len(dict[sensor]), color=colors[i])
    y_label.append(sensor)
    ax.text(l1[i], i, 'Sensor ID: ' + str(sensor), fontsize=12)



lall = []
for i in range(0,23):
    if len(str(i)) == 1:
        lall.append('0'+str(i)+":00")
        continue
    lall.append( str(i) + ":00")

plt.xlim(0, 23)
plt.xticks(range(0, 23), lall, rotation=45)
plt.yticks([])
plt.xlabel('Hours in a day')

plt.show()



'''

fig.suptitle('The Distribution of Sensor Readings for London Motorway M25 in Jan. 2017', fontsize=20)
red_patch = mpatches.Patch(color='red', label='First sensor assigned')
green_patch = mpatches.Patch(color='green', label='Second sensor assigned')
blue_patch = mpatches.Patch(color='blue', label='Third sensor assigned')
plt.legend(handles=[red_patch, green_patch, blue_patch])
plt.yticks(range(len(os.listdir(path))), y, rotation=15, fontsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.yaxis.grid(True)
plt.show()
'''