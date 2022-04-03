import os

import pandas as pd
import numpy as np
import re
from collections import defaultdict
from data_structure.ReadScheme import *

def filter_redundancy(data, junctions, name='filtered_tweets'):
    counter = 0
    filtered_data = pd.DataFrame()
    for i in range(0, len(data.values)):
        print(i)
        for junction in junctions:
            if data.values[i, 1].lower().__contains__(junction):
                counter += 1
                filtered_data = filtered_data.append(data.iloc[i])

                break
    filtered_data.to_csv(name+'.csv')
    print('Relevant tweets number: ', len(filtered_data.values), ' out of ', len(data.values))

# Remove words @to_remove list from the given list.
def removeWords(extracted_words, to_remove):
    clean_words = extracted_words.copy()
    for word in extracted_words:
        for restrict in to_remove:
            if word.lower().__contains__(restrict.lower()):
                clean_words.remove(word)
                break

    return clean_words

def checkWeatherCondition(word): # 2
    return word.lower().__contains__('rain') or  word.lower().__contains__('snow')

def checkHumanReaction(word): # 3
    return word.lower().__contains__('patience') or word.__contains__('patient') or word.__contains__('sad')

def checkRoadEvent(word): # 22
    return word.lower().__contains__('congest') or word.lower().__contains__('jam') or word.lower().__contains__('delay')or\
                word.lower().__contains__('stop')or word.lower().__contains__('slow') or word.lower().__contains__('block')  or word.lower().__contains__('wait')\
                 or word.lower().__contains__('clos') or word.lower().__contains__('stuffed')\
                or word.lower().__contains__('lock')   or word.lower().__contains__('roadwork')  or word.lower().__contains__('full')\
                or word.lower().__contains__('crook') or word.lower().__contains__('heavy') or word.lower().__contains__('obstruct') \
                or word.lower().__contains__('busy') or word.lower().__contains__('stationary') or word.lower().__contains__('standstill')\
                or  word.lower().__contains__('busi') or  word.lower().__contains__('heavi')  or word.lower().__contains__('work')\
                or word.lower().__contains__('shut')

def checkAccident(word): # 2
    return word.lower().__contains__('accident') or word.lower().__contains__('incident')

def checkEventCaused(word): # 28
    return word.lower().__contains__('slip') or word.lower().__contains__('trap') or word.lower().__contains__('divert') or word.lower().__contains__('overturn')\
                or word.lower().__contains__('spillage') or word.lower().__contains__('crash') or  word.lower().__contains__('crane')\
                or word.lower().__contains__('explosion') or  word.lower().__contains__('fire') or word.lower().__contains__('burn')\
                or word.lower().__contains__('torch') or word.lower().__contains__('lift') or word.lower().__contains__('extinguish')\
                or word.lower().__contains__('stuck') or word.lower().__contains__('breakdown') or word.lower().__contains__('roll') \
                or word.lower().__contains__('damage') or word.lower().__contains__('down') or word.lower().__contains__('break') or word.lower().__contains__('broken') \
                or word.lower().__contains__('broke') or word.lower().__contains__('abnormal') or word.lower().__contains__('fallen') \
                or word.lower().__contains__('debris') or word.lower().__contains__('repair') or word.lower().__contains__('disrupt') \
                or word.lower().__contains__('collide') or word.lower().__contains__('collision')

def checkRoadCondition(word): # 11
    return word.lower().__contains__('injuries') or word.lower().__contains__('injury') or word.lower().__contains__('ambulance') \
                or word.lower().__contains__('smoke') or word.lower().__contains__('pain') or word.lower().__contains__('emergency')\
                or word.lower().__contains__('police')\
                or word.lower().__contains__('officer') or word.lower().__contains__('investigat') or word.lower().__contains__('run')\
                or word.lower().__contains__('barrier')

def checkWarning(word): # 5
    return  word.lower().__contains__('problem') or word.lower().__contains__('trouble') or word.lower().__contains__('issu') \
                or  word.lower().__contains__('warn')  or  word.lower().__contains__('caution')

def textCategorize(data):
    weather         = 0
    human_reaction  = 0
    road_events     = 0
    accident        = 0
    event           = 0
    condition_without_event = 0
    warning         = 0
    words = ['']
    for row in data.values:
        text = row[2]
        list_words = text.split(' ')
        # weather 2
        if checkWeatherCondition(text):
            weather+=1
        # Human reaction 2
        if checkHumanReaction(text):
            human_reaction+=1
        # Represent events on road 21
        if  checkRoadEvent(text):
            road_events+=1
        # Mention accident 2
        if checkAccident(text):
            accident += 1
        # Represent events on accidents 26
        if checkEventCaused(text):
            event += 1
        # Represent accident without the event itself 13
        if checkRoadCondition(text):
            condition_without_event += 1
        # Represent warning 5
        if checkWarning(text):
            warning += 1
        # Remove links
        words+= removeWords(list_words, to_remove=['https'])

    unique_words = set(words)

    print(unique_words)
    print('Number of unique words:',len(words))
    print('Number of cleaned words:',71)
    print('weather:',weather)
    print('human reaction:',human_reaction)
    print('road events:',road_events)
    print('accident:',accident)
    print('caused event:',event)
    print('road activity without event:',condition_without_event)
    print('warning:',warning)

def removeDublicate(data, name='removed_dublication'):
    dublicate = 0
    last_info = []
    curr_info = []
    out = pd.DataFrame()
    for i, time in enumerate(data['created_at']):
        time = pd.to_datetime(time)

        content     = data.values[i][2]
        publisher   = data.values[i][5]
        desc        = data.values[i][6]
        time_h      = time.hour
        time_m      = time.minute

        # Remove links
        if content.find('https') != -1:
            content = content[: content.find('https')]

        # Store info
        curr_info.append(content)
        curr_info.append(publisher)
        curr_info.append(desc)
        curr_info.append(time_h)
        curr_info.append(time_m)

        if len(last_info) == 0:
            last_info = curr_info.copy()
            continue

        dub = True

        if curr_info == last_info:
            dublicate += 1
            print('found', publisher)
            dub = False
        else:
            out = out.append(data.iloc[i-1])

        # Add last record
        if i+1 > len(data.values) and dub:
            out = out.append(data.iloc[i])

        last_info = curr_info.copy()
        curr_info.clear()

    print('Number of dublicated tweets is ', dublicate)
    out.to_csv(name + '.csv')

def removeDataWithMissingInfo(data, name = 'removedMissingData'):

    missing_records = 0
    out = pd.DataFrame()
    for i, content in enumerate(data['text']):
        if not content.lower().__contains__('clockwise') and \
                not  content.lower().__contains__('anticlockwise') and \
                not  content.lower().__contains__('anti-clockwise'):
            print('*: ', content)
            missing_records+=1
            continue
        out = out.append(data.iloc[i])
    print('Number of records with missing data: ', missing_records)
    out.to_csv(name + '.csv')

def generatePerJunction(data):

    nodes = os.listdir('../../output/')

    scheme = ReadScheme()
    relevant = 0
    count    = 0
    error    = 0
    out = defaultdict(list)
    output_file = pd.DataFrame()
    for k, content in enumerate(data['text']):

        content = ' '.join(removeWords(content.split(' '), ['https']))
        scheme.addContent(content)

        complete, causes, directions, connections, junctions = scheme.extractSchemes()
        if len(connections) > 0 and len(directions) > 0 and len(junctions) > 1:
            if directions[0] == 'anticlockwise' or directions[0].__contains__('-'):
                jun1_ = int(junctions[0][1:])
                jun2_ = int(junctions[1][1:])
                sor = sorted([jun1_, jun2_])
                for i in range(sor[0], sor[1]+1, 2) :
                    text = 'The M25 ' + (
                        directions[0] if not directions[0].__contains__('-') else 'anticlockwise') + ' between junctions ' \
                           + ('J'+str(i+1) + ' and ' + 'J'+str(i))
                    out[text].append(list(data.iloc[k].values) + [' '.join(causes)])
            else:
                jun1 = int(junctions[0][1:])
                jun2 = int(junctions[1][1:])
                sor = sorted([jun1, jun2])
                for i in range(sor[0], sor[1]+1, 2) :
                    text = 'The M25 ' + (
                        directions[0] if not directions[0].__contains__('-') else 'anticlockwise') + ' between junctions ' \
                           + ('J'+str(i) + ' and ' + 'J'+str(i+1))
                    out[text].append(list(data.iloc[k].values) + [' '.join(causes)])

            if len(junctions) > 3:
                if directions[0] == 'anticlockwise' or directions[0].__contains__('-'):
                    jun1 = int(junctions[2][1:])
                    jun2 = int(junctions[3][1:])
                    sor = sorted([jun1, jun2])
                    for i in range(sor[0], sor[1] + 1, 2):
                        text = 'The M25 ' + (
                            directions[0] if not directions[0].__contains__(
                                '-') else 'anticlockwise') + ' between junctions ' \
                               + ('J' + str(i+1) + ' and ' + 'J' + str(i))
                        out[text].append(list(data.iloc[k].values) + [' '.join(causes)])
                else:
                    jun1 = int(junctions[2][1:])
                    jun2 = int(junctions[3][1:])
                    sor = sorted([jun1, jun2])
                    for i in range(sor[0], sor[1] + 1, 2):
                        text = 'The M25 ' + (
                            directions[0] if not directions[0].__contains__(
                                '-') else 'anticlockwise') + ' between junctions ' \
                               + ('J' + str(i) + ' and ' + 'J' + str(i + 1))
                        out[text].append(list(data.iloc[k].values) + [' '.join(causes)])

                if len(junctions) == 3:
                    text = 'The M25 ' + (directions[0] if not directions[0].__contains__('-') else 'anticlockwise') \
                               + ' at junction ' + junctions[2]
                    out[text].append(list(data.iloc[k].values) + [' '.join(causes)])
        elif len(connections) > 0 and len(directions) > 0 and len(junctions) > 0:
            for junction in junctions:
                text = 'The M25 '+(directions[0] if not directions[0].__contains__('-') else 'anticlockwise') \
                           + ' at junction ' + junction
                out[text].append(list(data.iloc[k].values) + [' '.join(causes)])
        scheme.clear()

    for key in out:
        file_out = pd.DataFrame()
        print(len(out[key]))
        for value in out[key]:
            file_out = file_out.append(pd.DataFrame([value], columns=['10','10','10','tweet_id', 'text', 'created_at', 'user_id', 'user_name', 'user_description', 'followers_count', 'verified', 'cause']))
        file_out.to_csv('ext/'+key+'.csv')


def computeDuration():
    count = 0
    duration = 0
    for file in os.listdir('ext/'):
        sensor_file = file[:-4] + '  ' + file[-4:]
        if file.__contains__('between'):
            sensor_file =file[:-11] +' '+ file[-11:-4] +'  ' + file[-4:]

        sensor_data = pd.read_csv('../../output/' + sensor_file)
        tweet_data  = pd.read_csv('ext/'+file)
        for tweet_date in pd.to_datetime(tweet_data['created_at']):
            tweet_mints = tweet_date.day * 24 * 60 + tweet_date.hour * 60 + tweet_date.minute
            start       = False
            start_time  = -1
            end_time    = -1
            for i, sensor_date in enumerate(pd.to_datetime(sensor_data['date'])):
                sensor_mints = sensor_date.day * 24 * 60 + sensor_date.hour * 60 + sensor_date.minute
                if sensor_mints - tweet_mints < 15 and  sensor_mints - tweet_mints  >= 0 and start == False:
                    event = sensor_data['class'].iloc[i]
                    if int(event) == 1:
                        start_time = tweet_mints
                        start = True
                    else:
                        break
                if start:
                    event = sensor_data['class'].iloc[i]
                    if  int(event) == 0:
                        start = False
                        end_time = sensor_mints
                        duration += end_time - start_time
                        count+=1
                        print('average duration over time', duration/float(count))
                        break
    return duration/float(count)


def addGroundTruth():
    count = 0
    duration = 0
    for file in os.listdir('ext/'):
        sensor_file = file[:-4] + '  ' + file[-4:]
        if file.__contains__('between'):
            sensor_file = file[:-11] + ' ' + file[-11:-4] + '  ' + file[-4:]

        sensor_data = pd.read_csv('../../output/' + sensor_file)
        tweet_data = pd.read_csv('ext/' + file)
        for tweet_date in pd.to_datetime(tweet_data['created_at']):
            tweet_mints = tweet_date.day * 24 * 60 + tweet_date.hour * 60 + tweet_date.minute
            start = False
            start_time = -1
            end_time = -1
            for i, sensor_date in enumerate(pd.to_datetime(sensor_data['date'])):
                sensor_mints = sensor_date.day * 24 * 60 + sensor_date.hour * 60 + sensor_date.minute
                if sensor_mints - tweet_mints < 15 and sensor_mints - tweet_mints >= 0 and start == False:
                    event = sensor_data['class'].iloc[i]
                    if int(event) == 1:
                        start_time = tweet_mints
                        start = True
                    else:
                        break
                if start:
                    event = sensor_data['class'].iloc[i]
                    if int(event) == 0:
                        start = False
                        end_time = sensor_mints
                        duration += end_time - start_time
                        count += 1
                        print('average duration over time', duration / float(count))
                        break
    return duration / float(count)

data = pd.read_csv('C:/Users/moham/PycharmProjects/software_paper/tweets/analysis/output/removedMissingData.csv')

#junctions = ['j1A'] + ['j'+str(i) for i in range(2, 32)]

# Only get tweets contain junctions
#filter_redundancy(data, junctions)
# Identify categories.
#textCategorize(data)
# Remove dublicated (time in mints, publisher, tweet content)
#removeDublicate(data)
#removeDataWithMissingInfo(data)
# Tweets file for each location
#generatePerJunction(data)
# Compute duration, average.
avg_duration = computeDuration()
print('average in minutes', avg_duration)
# Matching (put words list corresponding each class

# Preprocessing for neural network.
# Training & Testing.


