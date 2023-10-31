# !/usr/bin/env python
# -*-coding:utf-8 -*-

import torch.nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import random
import pandas as pd


class MyDataset(Dataset):
    def __init__(self, data):
        super(MyDataset, self).__init__()
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def create_sequences(df, time, all_featuers, window_size, label, seq: list):
    df.sort_values(time, ascending=True, inplace=True, ignore_index=True)
    current_arr = df[all_featuers]
    risk_arr = df[label]
    for i in range(len(risk_arr)+1 - window_size):
        train_seq_arr = current_arr[i: i + window_size]
        train_seq = torch.FloatTensor(train_seq_arr.values)
        train_label = torch.FloatTensor([risk_arr[i + window_size-1]]).view(-1)
        seq.append((train_seq, train_label))


def get_dataloader(seq_data, data_id='', time='time', all_featuers=[], label='', window_size=7, bash_size=64, test_size=0.2, shuffer=True):

    seq = []
    seq_data.groupby([data_id]).apply(
        lambda x: create_sequences(df=x, time=time, all_featuers=all_featuers, window_size=window_size, label=label, seq=seq))
    random.seed(10)

    # 先对数据进行划分，再进行shuffer
    train_len = int(len(seq) * (1 - test_size))
    if shuffer is True:
        random.shuffle(seq)  # 避免过拟合，在此处进行shuffle

    Dtr = seq[0: train_len]
    Dte = seq[train_len:len(seq)]
    train = MyDataset(Dtr)
    test = MyDataset(Dte)
    Dtr = DataLoader(dataset=train, batch_size=bash_size, shuffle=False, num_workers=0)
    Dte = DataLoader(dataset=test, batch_size=bash_size, shuffle=False, num_workers=0)

    return Dtr, Dte


if __name__ == '__main__':
    data = pd.read_csv("../data/dataset.csv")
    data['ds'] = data[['YEAR', 'MO', 'DY', 'HR']].apply(lambda x: f"{x[0]}-{x[1]}-{x[2]} {x[3]}:00:00", axis=1)
    data['ds'] = pd.to_datetime(data['ds'])
    train_size = int(len(data) * 0.2)
    Dtr, Dte = get_dataloader(data, time='ds', label='WS10M', window_size=48, train_size=train_size)
    for x_train, y_train in Dtr:
        print(x_train.shape, y_train.shape)
        break
