import pandas as pd
import numpy as np
import os
from sklearn import svm

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from imblearn.pipeline import Pipeline
from keras.layers import Dropout,RandomFourierFeatures
from imblearn.under_sampling import RandomUnderSampler

from sklearn.metrics import precision_recall_fscore_support
from tensorflow import keras
from imblearn.over_sampling import SMOTE
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

def write_data(model):
    MAIN_PATH = 'groundtruth/'

    for file in os.listdir(MAIN_PATH):
        tweet_data = pd.read_csv(MAIN_PATH + file)

        one_hot_encoding = np.zeros((len(tweet_data), len(accepted_words)))
        for i, cause in enumerate(tweet_data['cause']):
            if type(cause) == str:
                for word in cause.split():
                    index = np.where(accepted_words == word)
                    one_hot_encoding[i, index] = 1

        X_train = np.asarray(one_hot_encoding).astype(np.float32)

        pred = model.predict(X_train)
        pred[pred >= 0.5]   = 1
        pred[pred < 0.5]    = 0

        data_pred = np.concatenate((tweet_data.values, X_train), axis=1)
        out = pd.DataFrame(data_pred[:,1:], columns=['tweet_id', 'text', 'created_at', 'user_id',
                                                                       'user_name', 'user_description',
                                                                       'followers_count', 'verified', 'cause',
                                                                       'class'] + ['f'+str(i) for i in range(len(X_train[0]))])
        out.to_csv('predicted_output_combin/'+ file)


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
y=y.astype('int')
print(y)
over = SMOTE()

over = SMOTE()
x, y = over.fit_resample(x, y)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=40)
print('ss',len(X_train) , np.sum(y_train))

print(np.sum(y_test))
print(np.sum(y_train))

X_train = np.asarray(X_train).astype(np.float32)
y_train = np.asarray(y_train).astype(np.float32)
X_test = np.asarray(X_test).astype(np.float32)
y_test = np.asarray(y_test).astype(np.float32)
# Model architecture

model = Sequential()
model.add(Dense(50, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))

model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


#model.fit(x, y, epochs=100, batch_size=5, validation_data=(X_test, y_test))
#model = keras.models.load_model('model2/')
# evaluate the keras model
_, accuracy = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy*100))
pred = model.predict(X_test)
pred[pred >= 0.5] = 1
pred[pred < 0.5] = 0
print(precision_recall_fscore_support(y_test, pred, average='macro'))
write_data(model)
#model.save('model_oversampling')
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
'SVM : ========================='
'''
clf = svm.SVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
'''
#===========================

