# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# Time       ：2022/8/10 9:46
# Author     ：tong
# Description：
"""
import numpy as np
import pandas as pd
from entity.dataloader import get_dataloader
from model.time_series import LSTM, GRU, RNN
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import torch
import gc
import joblib
from model import EpochEndCheckpoint

import pandas as pd
from entity.dataloader import get_dataloader
from model.time_series import LSTM, GRU, RNN
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import torch
import gc
import joblib
from model import EpochEndCheckpoint

path = "./data/trainnew1_1.csv"

def main():
    hidden_size = 32
    num_layers = 2
    output_size = 1
    batch_size = 12
    device = 'cpu'
    window_size = 1
    lr = 0.001
    epochs = 200
    test_size = 0.2
    data= pd.read_csv(path)
    data = pd.get_dummies(data, columns=['道路属性'])

    all_road_attrs = ['道路属性_主干路', '道路属性_次干路', '道路属性_快速路', '道路属性_支路']
    for attr in all_road_attrs:
        if attr not in data.columns:
            data[attr] = 0


    features = ['路龄', '车行道小修金额', '车行道零星金额', '车行道中修金额', '掘路修复金额', '前一年pci', '车道面积', '道路属性_主干路', '道路属性_次干路', '道路属性_快速路',
                '道路属性_支路']


    time = '年份'
    label = '当年pci'
    feature_minmax = MinMaxScaler()
    data[features] = feature_minmax.fit_transform(data[features])
    label_encoder = MinMaxScaler()
    data[label] = label_encoder.fit_transform(data[label].values.reshape(-1, 1))
    joblib.dump(feature_minmax, "./data/model/features_mimmax.pkl")
    joblib.dump(label_encoder, "./data/model/label_mimmax.pkl")
    Dtr, Dte = get_dataloader(data, data_id=id, time=time, all_featuers=features, label=label, window_size=window_size,
                              bash_size=batch_size, test_size=test_size, shuffer=False)

    model = GRU(len(features), hidden_size, num_layers, output_size, batch_size=batch_size, device=device,
                window_size=window_size).to(device)
    model.compile("adam", "mae",
                  metrics=['mae', 'rmse', 'mse'],
                  lr=lr)
    callbacks = [
        EpochEndCheckpoint(filepath="./data/model/net_model_RNN3", save_best_only=True,
                           mode='min',
                           verbose=1,
                           monitor='val_mae')
    ]



    train_loss_list, test_loss_list = model.fit(train=Dtr, batch_size=batch_size, epochs=epochs, validation_data=Dte, callbacks=callbacks)
    plt.plot(range(len(train_loss_list)), train_loss_list, c='r', label='train')
    plt.plot(range(len(test_loss_list)), test_loss_list, c='b', label='test')
    plt.legend(loc='upper right', fancybox=True, framealpha=1, shadow=True, borderpad=1)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()
    del Dtr, Dte
    gc.collect()
    # 在训练结束后保存模型权重
    torch.save(model.state_dict(), "./data/model/net_model_RNN3_weights.pt")  # 使用.pt作为文件扩展名，这是PyTorch模型权重的常用扩展名
if __name__ == '__main__':
    main()
