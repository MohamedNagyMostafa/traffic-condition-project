import numpy as np
import pandas as pd
import os
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt
def readAllFiles():
    out = pd.DataFrame(columns=['index','sensor','date','class','reason','speed','feat1','feat2','feat3',
                       'feat4','feat5','feat6','predicted','user_name', 'followers_count', 'verified'])
    for file in os.listdir('../final_output/'):
        file_path = '../final_output/'+file
        out = out.append(pd.read_csv(file_path))

    return out

data = readAllFiles()
datetime = pd.to_datetime(data['date'])
time_in_mints = []
for record in datetime:
    time_in_mints.append(record.hour*60 + record.minute)

pred_num = data['predicted'].values
pred = pred_num.copy()
pred_num[pred == 2] = 0
pred_num[pred == 1] = 1
pred_num[pred == 0] = 2

feature_data = np.concatenate((np.expand_dims(data['speed'].values, axis=1),
                               np.expand_dims(data['feat1'].values, axis=1),
                               np.expand_dims(data['feat2'].values, axis=1),
                               np.expand_dims(data['feat3'].values, axis=1),
                               np.expand_dims(data['feat4'].values, axis=1),
                               np.expand_dims(data['feat5'].values, axis=1),
                               np.expand_dims(data['feat6'].values, axis=1),
                               np.expand_dims(pred_num, axis=1),
                               np.expand_dims(data['class'].values, axis=1)), axis=1)

feature_dataframe = pd.DataFrame(feature_data, columns=['speed', 'feat1', 'feat2', 'feat3', 'feat4', 'feat5', 'feat6', 'predicted (Tweets)','class'])

plt.rcParams.update({'font.size': 16, 'xtick.labelsize': 12, 'ytick.labelsize': 16})

ax = sns.heatmap(feature_dataframe.astype(float).corr(),vmin=-1, vmax=1, annot=True)

plt.show()
