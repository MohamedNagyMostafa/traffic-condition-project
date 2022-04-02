import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.patches as mpatches
import seaborn as sns

def examineDurationPerLocation():
    path = 'output/'

    # handle y-axis label
    y = []
    for text in os.listdir(path):
        b = text[8:-5].split(' ')

        direction = b[0]
        region1 = b[3]
        if len(b) > 5:
            region2 = b[6]

            y.append(direction + " (" + region1 + " to " +region2 + ")")
            continue
        y.append(direction + " in " + region1)
    sns.set()
    fig, ax = plt.subplots()


    for k, loc in enumerate(os.listdir(path)):
        data = pd.read_csv(path + loc)
        pre = -1
        durations = []
        data['date'] = data['date']

        sensors = []

        link = defaultdict(list)

        for i, dateTime in enumerate(pd.to_datetime(data['date'])):
            if i == 0:
                begin_date = dateTime

            if i+1 == len(data['date'].values):
                end_date = dateTime

            link[data['sensor'].values[i]].append((dateTime.timestamp() - pd.to_datetime('2017-01-01 00:00:00').timestamp())/(60*60*24))

            if pre == -1:
                pre = dateTime.timestamp()
                continue

            durations.append((dateTime.timestamp() - pre))
            pre = dateTime.timestamp()

        print(loc, ' has diff duration', len(set(durations)))
        print('duration are ', set(durations))
        print('begin date', begin_date)
        print('end date', end_date)
        print('number of records', len(data['date']))
        print('========================')

        #ploting
        colors = ['r', 'g', 'b']

        for i, sensor in enumerate(link.keys()):

            plt.scatter(link[sensor],[k] * len(link[sensor]) , color =colors[i], alpha=0.3, s = 8)
            plt.xlabel('Readings duration over 30 days')
            plt.xlim(1, 30)
            plt.xticks(range(1, 31), range(1, 31), rotation=0)
    #fig.suptitle('The Distribution of Sensor Readings for London Motorway M25 in Jan. 2017', fontsize=20)
    red_patch = mpatches.Patch(color='red', label='First sensor assigned')
    green_patch = mpatches.Patch(color='green', label='Second sensor assigned')
    blue_patch = mpatches.Patch(color='blue', label='Third sensor assigned')
    plt.legend(handles=[red_patch, green_patch, blue_patch])
    plt.yticks(range(len(os.listdir(path))),y, rotation=15, fontsize = 8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.yaxis.grid(True)
    plt.show()



def getUniqueDescription(data):
    desc_column  = data['description']

    locations = []
    total_sensors = 0
    # extract all locations
    for i, value in enumerate(desc_column.values):
        loc = value[:value.find(', Reason')]
        locations.append(loc)

    locations_unique = set(locations)
    locations = np.array(locations)

    # Sensor/s per location
    for location in locations_unique:
        indices = np.where(locations == location)

        classes  = []
        sensors  = []
        dates    = []
        reasons  = []
        speeds   = []
        feat1s   = []
        feat2s   = []
        feat3s   = []
        feat4s   = []
        feat5s   = []
        feat6s   = []

        for index in indices[0]:
            sensor = data['SensorsID'].values[index]
            date   = data['date'].values[index]
            reason = data['description'].values[index][data['description'].values[index].find(', Reason') + len(', Reason')+3:]
            speed  = data['speed'].values[index]
            feat1  = data['Unnamed: 7'].values[index]
            feat2  = data['Unnamed: 8'].values[index]
            feat3  = data['Unnamed: 9'].values[index]
            feat4  = data['Unnamed: 10'].values[index]
            feat5  = data['Unnamed: 11'].values[index]
            feat6  = data['Unnamed: 12'].values[index]
            class_ = data['class'].values[index]

            sensors.append(sensor)
            dates.append(date)
            reasons.append(reason)
            speeds.append(speed)
            feat1s.append(feat1)
            feat2s.append(feat2)
            feat3s.append(feat3)
            feat4s.append(feat4)
            feat5s.append(feat5)
            feat6s.append(feat6)
            classes.append(class_)

        out = pd.DataFrame({'sensor': sensors, 'date': dates, 'class':classes,
                      'reason': reasons, 'speed': speeds,
                      'feat1':feat1s, 'feat2':feat2s,
                      'feat3':feat3s, 'feat4':feat4s,
                      'feat5':feat5s, 'feat6':feat6s})

        out.to_csv('output/'+location[11:]+".csv")


        total_sensors += len(set(sensors))
        if len(set(sensors)) == 1:
            print('Only one sensor: ', set(sensors), ' loc: ', location)
        else:
            print('Total number of sensors are:', len(set(sensors)), ' are ', set(sensors), ' loc: ', location)

    print('total sensors ', total_sensors, ' vs actual :', len(set(data['SensorsID'].values)))
# Notes:
# Data has big variations between readings, max duration is 10 days.

data = pd.read_csv('dataset/data.csv')
print(len(data['SensorsID'].tolist()))
'''
datetime = pd.to_datetime(data['Timestamp'], unit='s')
print('number of sensors', len(set(data['SensorsID'].tolist())))
duration_diff = []
duration_diff_without_neg = []
errors_loc = []
counter = defaultdict(int)
detail = defaultdict(str)
len_lessthan_fMints = 0

for seconds in range(1, data['Timestamp'].size):
    try:
        presecond = int(data['Timestamp'].values[seconds-1])
    except:
        errors_loc.append(seconds-1)
        continue
    try:
        currsecond = int(data['Timestamp'].values[seconds])
    except:
        errors_loc.append(seconds-1)
        continue

    duration_diff.append((currsecond - presecond)/60)
    if currsecond - presecond > 0:
        duration_diff_without_neg.append((currsecond - presecond)/60)
        len_lessthan_fMints += 1 if (currsecond - presecond)/60 <= 5 else 0

    counter[str((currsecond - presecond)/60)] += 1

print('errors index', set(errors_loc))
print('unique diff', set(np.abs(duration_diff)), 'unique durations in minutes')
print('max diff', max(np.abs(duration_diff))/(60*24) , 'days')
print('Mean  without negative (Minute):', np.mean(duration_diff_without_neg))
print('Standard Deviation without negative(Minute):', np.std(duration_diff_without_neg))
print('Median  without negative(Minute):', np.median(duration_diff_without_neg))
print('Number of records less than 5 mintues', len_lessthan_fMints)
print('Number of records more than 5 mintues', len(data.values) - len_lessthan_fMints)
print('total records', len(data.values))
print('========== Display durations, records count ==========')

for key in sorted(counter, key=counter.get, reverse=True):
    print('duration', key,'records', counter.get(key))



#plt.plot(range(len(duration_diff)), duration_diff)
#plt.xlabel('records in sequence')
#plt.ylabel('duration difference')
#plt.show()

'''