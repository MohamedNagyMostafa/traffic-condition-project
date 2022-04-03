import pandas as pd
import numpy as np
import os

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.metrics import precision_recall_fscore_support
from tensorflow import keras
from keras.wrappers.scikit_learn import KerasClassifier

accepted_words = np.array(['rain','snow','patience','patient','sad','conges','jam','delay','stop','slow','block','wait', 'queu', 'flood','hard','abnormal',
                      'clos','stuffed','lock','roadwk','full','crook','heavy','obstruct','busy','stationary','standstill',
                      'busi','heavi','shut','accident','incident','slip','trap','divert','overturn','spillage','crash','crane',
                      'explosion','fire','burn','tch','lift','extinguish','stuck','breakdown','roll','damage','down','break','broken',
                      'broke','abnmal','fallen','debris','repair','disrupt','collide','collision','injuries','injury','ambulance','smoke',
                      'pain','emergency','police','officer','investigat','work','run','barrier','problem','trouble','issu','warn','caution'])


def read_data():
    MAIN_PATH = 'groundtruth/'

    data = np.array([])

    for file in os.listdir(MAIN_PATH):
        tweet_data = pd.read_csv(MAIN_PATH + file)
        if len(data) == 0:
            data = tweet_data.values
        else:
            data = np.concatenate((data, tweet_data), axis=0)

    return data


data = read_data()
total_records = len(data)
print('total records is ', total_records)

# shuffle data
print('before shuffle, first ten records', data[:30, -1])
indices = np.array(range(0, total_records))
np.random.shuffle(indices)
data = data[indices]
print('After shuffle, first ten records', data[:30, -1])
# Data preperation
data = data[:,-2:]
one_hot_encoding = np.zeros((len(data), len(accepted_words)))
for i, cause in enumerate(data[:,0]):
    if type(cause) == str:
        words = cause.split()
        for word in words:
            index = np.where(accepted_words == word)
            one_hot_encoding[i, index] = 1

x = one_hot_encoding
y = data[:, -1]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=40)
print(np.sum(y_test))
print(np.sum(y_train))

X_train = np.asarray(X_train).astype(np.float32)
y_train = np.asarray(y_train).astype(np.float32)
X_test = np.asarray(X_test).astype(np.float32)
y_test = np.asarray(y_test).astype(np.float32)
# Model architecture

model = Sequential()
model.add(Dense(200, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])


model.fit(X_train, y_train, epochs=500, batch_size=10, validation_data=(X_test, y_test))
model.save('model/')
# evaluate the keras model
_, accuracy = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy*100))
pred = model.predict(X_test)
pred[pred >= 0.5] = 1
pred[pred < 0.5] = 0
print(precision_recall_fscore_support(y_test, pred, average='macro'))

