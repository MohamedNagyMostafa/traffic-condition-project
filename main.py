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


class CNNLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)
        #self.conv4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=1)
        self.rnn1 = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.rnn2 = nn.LSTM(hidden_dim, hidden_dim, layer_dim, bidirectional=False, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.batch_size = None
        self.hidden = None

    def forward(self, x):
        #print('x: ', x.shape)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        #x = F.relu(self.conv4(x))
        out, (h1, c1) = self.rnn1(x)

        #h0, c0 = self.init_hidden(x)
        #out, (h1, c1) = self.rnn1(x, (h0, c0))
        out, (h2, c2) = self.rnn2(out, (h1, c1))
        #out, (_, _) = self.rnn3(out, (h2, c2))
        #print('out: ', out.shape)
        out = self.fc(out[:, -1, :])
        return out

    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        return [t.cuda() for t in (h0, c0)]


class LSTMClassifier(nn.Module):
    """Very simple implementation of LSTM-based time-series classifier."""

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn1 = nn.LSTM(input_dim, hidden_dim, layer_dim, bidirectional=True, batch_first=True)
        self.rnn2 = nn.LSTM(2*hidden_dim, hidden_dim, layer_dim, bidirectional=True, batch_first=True)
        self.rnn3 = nn.LSTM(2*hidden_dim, hidden_dim, layer_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2*hidden_dim, output_dim)
        self.batch_size = None
        self.hidden = None

    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        out, (h1, c1) = self.rnn1(x, (h0, c0))
        out, (h2, c2) = self.rnn2(out, (h1, c1))
        out, (_, _) = self.rnn3(out, (h2, c2))
        #print('out: ', out.shape)
        out = self.fc(out[:, -1, :])
        return out

    def init_hidden(self, x):
        h0 = torch.zeros(2*self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(2*self.layer_dim, x.size(0), self.hidden_dim)
        return [t.cuda() for t in (h0, c0)]


def readData():
    files = np.array(os.listdir('final_output_span_early_fusion/'))
    # np.random.shuffle(files)
    files = np.load('file_indices.npy')
    seperated_readings = None
    # np.save('file_indices.npy', files)
    for file in files:
        data = pd.read_csv('final_output_span_early_fusion/' + file)

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


features = ['class', 'speed', 'feat3', 'date']

# Add the features from one-hot encoding
for f in range(0, 77):
    features.append('f' + str(f))

model_name = 'ls_ef_best_bi_lstm_v2.pth'

input_dim = len(features) - 1
hidden_dim = 256
layer_dim = 3
output_dim = 1
batch_size = 128

lr = 0.0001
n_epochs = 1000
min_loss = 1000000000000000
patience, trials = 50, 0

seperated_readings = readData()

print(seperated_readings.shape)

seperated_readings = convertTimeToMints(seperated_readings)

X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(seperated_readings[features], 'class', 0.2)

print(X_train.shape)
print(X_val.shape)
print(X_test.shape)

normalizeFeatures(X_train, features[1:4])
normalizeFeatures(X_val, features[1:4])
normalizeFeatures(X_test, features[1:4])

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
'''
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
    val_loss = 0
    cnt = 0
    for x_val, y_val in val_loader:
        x_val, y_val = x_val.view([batch_size, -1, input_dim]).cuda(), y_val.cuda()
        out = model(x_val)
        preds = torch.round(torch.sigmoid(out))

        val_loss += criterion(out, y_val)
        cnt += 1

        total += y_val.size(0)
        correct += (preds == y_val).sum().item()

    acc = correct / total
    val_loss /= cnt


    if epoch % 5 == 0:
        print(f'Epoch: {epoch:3d}. Loss: {loss.item():.4f}. Acc.: {acc:2.2%}')

    if min_loss - val_loss > 0.0001:
        trials = 0
        min_loss = val_loss
        torch.save(model.state_dict(), model_name)
        print(f'Epoch {epoch} best model saved with accuracy: {acc:2.2%} and loss: {min_loss:2.4}')
    else:
        trials += 1
        if trials >= patience:
            print(f'Early stopping on epoch {epoch}')
            break

'''
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score

model = LSTMClassifier(input_dim, hidden_dim, layer_dim, output_dim)
model.load_state_dict(torch.load(model_name))
model.cuda()
model.eval()

total_preds = None
total_test = None

for x_val, y_val in test_loader:
    x_val, y_val = x_val.view([batch_size, -1, input_dim]).cuda(), y_val.cuda()
    out = model(x_val)
    preds = torch.round(F.sigmoid(out))

    if total_preds is None:
        total_preds = preds
    else:
        total_preds = torch.cat((total_preds, preds), 0)

    if total_test is None:
        total_test = y_val
    else:
        total_test = torch.cat((total_test, y_val), 0)

    #total += y_val.size(0)
    #correct += (preds == y_val).sum().item()

    #acc = correct / total

    #print(f'Epoch:Acc Test.: {acc:2.2%}')

print(total_preds.shape)
print(total_test.shape)

total_preds = total_preds.detach().cpu().numpy()
total_test = total_test.detach().cpu().numpy()

print(accuracy_score(total_test, total_preds),end=',')
print(precision_score(total_test, total_preds),end=',')
print(recall_score(total_test, total_preds),end=',')
print(f1_score(total_test, total_preds),end=',')
print(roc_auc_score(total_test, total_preds))

#np.savetxt(model_name.split('.')[0] + '.csv', total_preds, delimiter=",")
np.savetxt('test_data.csv', total_test, delimiter=",")


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
