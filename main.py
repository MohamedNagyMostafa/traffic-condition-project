import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import torch.optim as optim


class LSTMClassifier(nn.Module):
    """Very simple implementation of LSTM-based time-series classifier."""

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.batch_size = None
        self.hidden = None

    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        out, (hn, cn) = self.rnn(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        return [t.cuda() for t in (h0, c0)]

def readData():
    files = np.array(os.listdir('final_output/'))
    #np.random.shuffle(files)
    files = np.load('file_indices.npy')
    seperated_readings = None
    #np.save('file_indices.npy', files)
    for file in files:
        data = pd.read_csv('final_output/' + file)

        if seperated_readings is None:
            seperated_readings = data
        else:
            seperated_readings = seperated_readings.append(data)

    return seperated_readings

def convertTimeToMints(seperated_readings):
    # Convert time to mins
    for i, date_row in enumerate(seperated_readings['date']):
        time = pd.to_datetime(date_row)
        new = time.hour * 60 + time.minute
        seperated_readings['date'].iloc[i] = new

    return seperated_readings

def normalizeFeatures(seperated_readings, features):

    for feature in features:
        seperated_readings[feature] = (seperated_readings[feature] - np.min(seperated_readings[feature])) / (
                    np.max(seperated_readings[feature]) - np.min(seperated_readings[feature]))

    return seperated_readings


def train_val_test_split(df, target_col, test_ratio):
    val_ratio = test_ratio / (1 - test_ratio)
    X, y = feature_label_split(df, target_col)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, shuffle=False)
    return X_train, X_val, X_test, y_train, y_val, y_test

def feature_label_split(df, target_col):
    y = df[[target_col]]
    X = df.drop(columns=[target_col])
    return X, y

input_dim = 3
hidden_dim = 256
layer_dim = 3
output_dim = 1
batch_size = 64

lr = 0.0005
n_epochs = 1000
best_acc = 0
patience, trials = 300, 0



seperated_readings = readData()

seperated_readings = convertTimeToMints(seperated_readings)

seperated_readings = normalizeFeatures(seperated_readings, ['speed', 'feat1','feat2','feat4','feat5','feat6','feat3', 'date'])

# remove irrelevant columns
features = ['class', 'speed','feat3' ,'date']

X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(seperated_readings[features], 'class', 0.2)

train_features = torch.from_numpy(X_train.values.astype('float32'))
train_targets = torch.from_numpy(y_train.values.astype('float32'))
val_features = torch.from_numpy(X_val.values.astype('float32'))
val_targets = torch.from_numpy(y_val.values.astype('float32'))
test_features = torch.from_numpy(X_test.values.astype('float32'))
test_targets = torch.from_numpy(y_test.values.astype('float32'))


train = TensorDataset(train_features, train_targets)
val = TensorDataset(val_features, val_targets)
test = TensorDataset(test_features, test_targets)

train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader_one = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)

iterations_per_epoch = len(train_loader)

model = LSTMClassifier(input_dim, hidden_dim, layer_dim, output_dim)
model = model.cuda()
criterion = nn.BCEWithLogitsLoss()
opt = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(1, n_epochs + 1):

    for i, (x_batch, y_batch) in enumerate(train_loader):
        model.train()
        x_batch = x_batch.view([batch_size, -1, input_dim]).cuda()
        y_batch = y_batch.cuda()
        opt.zero_grad()
        out = model(x_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        opt.step()

    model.eval()
    correct, total = 0, 0
    for x_val, y_val in val_loader:
        x_val, y_val = x_val.view([batch_size, -1, input_dim]).cuda(), y_val.cuda()
        out = model(x_val)
        preds = torch.round(F.sigmoid(out))
        total += y_val.size(0)
        correct += (preds == y_val).sum().item()

    acc = correct / total

    if epoch % 5 == 0:
        print(f'Epoch: {epoch:3d}. Loss: {loss.item():.4f}. Acc.: {acc:2.2%}')

    if acc > best_acc:
        trials = 0
        best_acc = acc
        torch.save(model.state_dict(), 'best.pth')
        print(f'Epoch {epoch} best model saved with accuracy: {best_acc:2.2%}')
    else:
        trials += 1
        if trials >= patience:
            print(f'Early stopping on epoch {epoch}')
            break

    correct, total = 0, 0

    for x_val, y_val in test_loader:
        x_val, y_val = x_val.view([batch_size, -1, input_dim]).cuda(), y_val.cuda()
        out = model(x_val)
        preds = torch.round(F.sigmoid(out))


        total += y_val.size(0)
        correct += (preds == y_val).sum().item()


    acc = correct / total
    print(f'Epoch:Acc Test.: {acc:2.2%}')
model.eval()


'''
activeBefore = True
activeWhile = False
activeAfter = False
data_collection = []
for reading in seperated_readings:
    before = 0
    while_ = 0
    after  = 0
    data_picker = []
    activeBefore = True
    activeWhile = False
    activeAfter = False
    for i, rec in enumerate(reading['class']):
        if rec == 0 and activeBefore == True:
            before+=1
            data_picker.append(reading.iloc[i])
        elif rec == 1 and activeBefore == True:
            activeBefore = False
            while_ += 1
            activeWhile = True
            data_picker.append(reading.iloc[i])
        elif rec == 1 and activeWhile == True:
            while_ +=1
            data_picker.append(reading.iloc[i])
        elif rec == 0 and activeWhile == True:
            activeWhile = False
            activeAfter = True
            after +=1
            data_picker.append(reading.iloc[i])
        elif rec == 0 and activeAfter == True:
            after+=1
            data_picker.append(reading.iloc[i])
        elif rec == 1 and activeAfter == True:
            print('turning to another reading zeros while before', before, ' while ', while_ , ' after ', after, ' total ', before + while_ + after)
            data_picker_cp = data_picker[-while_:].copy()
            activeAfter = False
            while_= 1
            after = 0
            before = len(data_picker_cp)
            activeWhile = True
            data_collection.append(data_picker)
            data_picker = data_picker_cp
            data_picker.append(reading.iloc[i])
        else:
            print('problem')
    data_collection.append(data_picker)


    print('File has before ', before, ' while ', while_ , ' after ', after, ' total ', before + while_ + after)

for collection in data_collection:
    print(len(collection))

'''