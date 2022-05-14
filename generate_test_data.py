import numpy as np
import os
import pandas as pd

from sklearn.model_selection import train_test_split

def readData():
    files = np.array(os.listdir('traffic-condition-project-master/final_output/'))
    #np.random.shuffle(files)
    files = np.load('file_indices.npy')
    seperated_readings = None
    #np.save('file_indices.npy', files)
    for file in files:
        data = pd.read_csv('traffic-condition-project-master/final_output/' + file)

        if seperated_readings is None:
            seperated_readings = data
        else:
            seperated_readings = seperated_readings.append(data)

    return seperated_readings

def feature_label_split(df, target_col):
    y = df[[target_col]]
    X = df.drop(columns=[target_col])
    return X, y

def train_val_test_split(df, target_col, test_ratio):
    val_ratio = test_ratio / (1 - test_ratio)
    X, y = feature_label_split(df, target_col)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, shuffle=False)
    return X_train, X_val, X_test, y_train, y_val, y_test


def convertTimeToMints(seperated_readings):
    # Convert time to mins
    for i, date_row in enumerate(seperated_readings['date']):
        time = pd.to_datetime(date_row)
        new = time.hour * 60 + time.minute
        seperated_readings['date'].iloc[i] = new

    return seperated_readings



seperated_readings = readData()

#seperated_readings = convertTimeToMints(seperated_readings)

#seperated_readings = normalizeFeatures(seperated_readings, ['speed', 'feat1','feat2','feat4','feat5','feat6','feat3', 'date'])

# remove irrelevant columns
features = ['class', 'date']

X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(seperated_readings[features], 'class', 0.2)

X_test = convertTimeToMints(X_test)

X_test = pd.concat((X_test, y_test), axis=1, ignore_index=True)

print(X_test.shape)

np.savetxt('test_data2.csv', X_test.to_numpy(), delimiter=',')