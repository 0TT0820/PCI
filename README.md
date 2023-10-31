import pandas as pd
from entity.dataloader import get_dataloader
from model.time_series import LSTM, GRU, RNN
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import torch
import gc
import joblib
from model import EpochEndCheckpoint

path = "./data/train5.csv"

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
    data = pd.read_csv(path)
data = pd.get_dummies(data, columns=['道路属性'])

    features = ['路龄', '当年pci','前一年pci','车道面积', '道路属性_主干路', '道路属性_次干路', '道路属性_快速路','道路属性_支路']
    id = '道路名称'
    time = '年份'
    label = '总钱数'
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


在RNN3模型的基础上增加数据（add）
import os
import pandas as pd
import torch
import gc
import joblib
from entity.dataloader import get_dataloader
from model.time_series import GRU
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from model import EpochEndCheckpoint


def main():


    # 数据处理
    path_new_data = "./data/trainnew2.csv"  # 新数据的路径
    hidden_size = 32
    num_layers = 2
    output_size = 1
    batch_size = 12
    device = 'cpu'
    window_size = 1
    lr = 0.001
    epochs = 200
    test_size = 0.2
    data= pd.read_csv(path_new_data)
    data = pd.get_dummies(data, columns=['道路属性'])
    features = ['路龄', '当年pci','前一年pci','车道面积', '道路属性_主干路', '道路属性_次干路', '道路属性_快速路','道路属性_支路']
    id = '道路名称'
    time = '年份'
    label = '总钱数'
    feature_minmax = MinMaxScaler()
    data[features] = feature_minmax.fit_transform(data[features])
    label_encoder = MinMaxScaler()
    data[label] = label_encoder.fit_transform(data[label].values.reshape(-1, 1))

    # 使用之前的scalers进行数据标准化
    joblib.dump(feature_minmax, "./data/model/features_mimmax.pkl")
    joblib.dump(label_encoder, "./data/model/label_mimmax.pkl")



    # 检查RNN3模型权重文件是否存在并加载
    # 设定权重文件路径
    checkpoint_path = "./data/model/net_model_RNN3_weights.pt"

    # 创建模型实例
    model = GRU(len(features), hidden_size, num_layers, output_size, batch_size=batch_size, device=device,
                window_size=window_size).to(device)

    # 模型编译
    model.compile("adam", "mae", metrics=['mae', 'rmse', 'mse'], lr=lr)

    # 如果权重文件存在，加载权重
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print("Loaded model weights from", checkpoint_path)

    Dtr, Dte = get_dataloader(data, data_id=id, time=time, all_featuers=features, label=label, window_size=window_size,
                              bash_size=batch_size, test_size=test_size, shuffer=False)

    # 模型训练
    callbacks = [
        EpochEndCheckpoint(filepath="./data/model/net_model_RNNnew3", save_best_only=True,
                           mode='min',
                           verbose=1,
                           monitor='val_mae')
    ]
    train_loss_list, test_loss_list = model.fit(train=Dtr, batch_size=batch_size, epochs=epochs, validation_data=Dte, callbacks=callbacks)

    # 可视化训练损失
    plt.plot(range(len(train_loss_list)), train_loss_list, c='r', label='train')
    plt.plot(range(len(test_loss_list)), test_loss_list, c='b', label='test')
    plt.legend(loc='upper right', fancybox=True, framealpha=1, shadow=True, borderpad=1)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()

    # 在训练结束后保存模型权重
    torch.save(model.state_dict(), "./data/model/net_model_RNNnew3_weights.pt")  # 使用.pt作为文件扩展名，这是PyTorch模型权重的常用扩展名
    del Dtr, Dte
    gc.collect()


if __name__ == '__main__':
    main()


调用模型进行预测：(rnn_predict)
import pandas as pd
import torch
import numpy as np
from entity.dataloader import get_dataloader
import joblib
id = '道路名称'
time = '年份'
label = '总钱数'
batch_size = 12
window_size = 1
test_path = "./data/predict_true2023_2.csv"
data = pd.read_csv(test_path)

data = pd.get_dummies(data, columns=['道路属性'])
    features = ['路龄', '当年pci','前一年pci','车道面积', '道路属性_主干路', '道路属性_次干路', '道路属性_快速路','道路属性_支路']
feature_minmax = joblib.load("./data/model/features_mimmax.pkl")

label_encoder = joblib.load("./data/model/label_mimmax.pkl")
data[features] = feature_minmax.transform(data[features])
data[label] = label_encoder.transform(data[label].values.reshape(-1, 1))
model = torch.load("./data/model/net_model_RNN3")

df_list = []

def get_score(x):
    x = x.sort_values(by=[time], ascending=True, ignore_index=True)
    last_data = x[['道路名称'] + [time] + features].values[-1]
    x = x[features].values[-3:]
    x = torch.tensor([x], dtype=torch.float)
    y = model(x)
    columns = ['道路名称'] + [time] + features + ['pred']
    tmp_df = pd.DataFrame(np.hstack([last_data, y.data.numpy()[0][0]]).reshape(1, -1), columns=columns)
    df_list.append(tmp_df)


data.groupby(by=id).apply(lambda x: get_score(x)).reset_index()
df = pd.concat(df_list)
df['pred'] = label_encoder.inverse_transform(df['pred'].values.reshape(-1, 1))
df[features] = feature_minmax.inverse_transform(df[features])

pred = np.array(df['pred'])
sum_money= sum(pred)/len(pred)

df.to_csv("./data/predict_res_2023_money.csv", index=False, encoding='utf-8')

print("好了")

预测PCI
训练模型(main)
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
    num_layers = 4
    output_size = 1
    batch_size = 12
    device = 'cpu'
    window_size = 1
    lr = 0.001
    epochs = 200
    test_size = 0.2
    data = pd.read_csv(path)
    data = pd.get_dummies(data, columns=['道路属性'])

    features = ['路龄', '车行道小修金额', '车行道零星金额', '车行道中修金额', '掘路修复金额', '前一年pci', '车道面积', '道路属性_主干路', '道路属性_次干路', '道路属性_快速路', '道路属性_支路']

    id = '道路名称'
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
        EpochEndCheckpoint(filepath="./data/model/net_model_RNN6", save_best_only=True,
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
torch.save(model.state_dict(), "./data/model/net_model_RNN6_weights.pt")  # 使用.pt作为文件扩展名，这是PyTorch模型权重的常用扩展名


if __name__ == '__main__':
    main()


在RNN3模型的基础上增加数据（add）
import os
import pandas as pd
import torch
import gc
import joblib
from entity.dataloader import get_dataloader
from model.time_series import GRU
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from model import EpochEndCheckpoint


def main():
    # Hyperparameters and configurations
    # ... [略去了一些参数设置，因为它们应该与你之前的代码相同]

    # 数据处理
    path_new_data = "./data/trainnew2.csv"  # 新数据的路径
    hidden_size = 32
    num_layers = 2
    output_size = 1
    batch_size = 12
    device = 'cpu'
    window_size = 1
    lr = 0.001
    epochs = 200
    test_size = 0.2
    data= pd.read_csv(path_new_data)
    data = pd.get_dummies(data, columns=['道路属性'])
    features = ['路龄', '车行道小修金额', '车行道零星金额', '车行道中修金额', '掘路修复金额', '前一年pci', '车道面积', '道路属性_主干路', '道路属性_次干路', '道路属性_快速路',
                '道路属性_支路']

    time = '年份'
    label = '当年pci'
    feature_minmax = MinMaxScaler()
    data[features] = feature_minmax.fit_transform(data[features])
    label_encoder = MinMaxScaler()
    data[label] = label_encoder.fit_transform(data[label].values.reshape(-1, 1))

    # 使用之前的scalers进行数据标准化
    joblib.dump(feature_minmax, "./data/model/features_mimmax.pkl")
    joblib.dump(label_encoder, "./data/model/label_mimmax.pkl")



    # 检查RNN6模型权重文件是否存在并加载
    # 设定权重文件路径
    checkpoint_path = "./data/model/net_model_RNN6_weights.pt"

    # 创建模型实例
    model = GRU(len(features), hidden_size, num_layers, output_size, batch_size=batch_size, device=device,
                window_size=window_size).to(device)

    # 模型编译
    model.compile("adam", "mae", metrics=['mae', 'rmse', 'mse'], lr=lr)

    # 如果权重文件存在，加载权重
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print("Loaded model weights from", checkpoint_path)

    Dtr, Dte = get_dataloader(data, data_id=id, time=time, all_featuers=features, label=label, window_size=window_size,
                              bash_size=batch_size, test_size=test_size, shuffer=False)

    # 模型训练
    callbacks = [
        EpochEndCheckpoint(filepath="./data/model/net_model_RNNnew6", save_best_only=True,
                           mode='min',
                           verbose=1,
                           monitor='val_mae')
    ]
    train_loss_list, test_loss_list = model.fit(train=Dtr, batch_size=batch_size, epochs=epochs, validation_data=Dte, callbacks=callbacks)

    # 可视化训练损失
    plt.plot(range(len(train_loss_list)), train_loss_list, c='r', label='train')
    plt.plot(range(len(test_loss_list)), test_loss_list, c='b', label='test')
    plt.legend(loc='upper right', fancybox=True, framealpha=1, shadow=True, borderpad=1)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()

    # 在训练结束后保存模型权重
    torch.save(model.state_dict(), "./data/model/net_model_RNNnew6_weights.pt")  # 使用.pt作为文件扩展名，这是PyTorch模型权重的常用扩展名
    del Dtr, Dte
    gc.collect()


if __name__ == '__main__':
    main()

调用模型进行预测,(rnn_predict)

import pandas as pd
import torch
import numpy as np
from entity.dataloader import get_dataloader
import joblib

id = '道路名称'
time = '年份'
label = '当年pci'
batch_size = 12
window_size = 1
test_path = "./data/predict2023.csv"
data = pd.read_csv(test_path)

data = pd.get_dummies(data, columns=['道路属性'])

features = ['路龄', '车行道小修金额', '车行道零星金额', '车行道中修金额', '掘路修复金额', '前一年pci', '车道面积', '道路属性_主干路', '道路属性_次干路', '道路属性_快速路',
            '道路属性_支路']


feature_minmax = joblib.load("./data/model/features_mimmax.pkl")
label_encoder = joblib.load("./data/model/label_mimmax.pkl")
data[features] = feature_minmax.transform(data[features])
data[label] = label_encoder.transform(data[label].values.reshape(-1, 1))
model = torch.load("./data/model/net_model_RNN3")

df_list = []

def get_score(x):
    x = x.sort_values(by=[time], ascending=True, ignore_index=True)
    last_data = x[['道路名称'] + [time] + features + ['道路名称2']].values[-1]
    x = x[features].values[-3:]
    x = torch.tensor([x], dtype=torch.float)
    y = model(x)
    columns = ['道路名称'] + [time] + features + ['道路名称2'] +  ['pred']
    tmp_df = pd.DataFrame(np.hstack([last_data, y.data.numpy()[0][0]]).reshape(1, -1), columns=columns)
    df_list.append(tmp_df)


data.groupby(by=id).apply(lambda x: get_score(x)).reset_index()
df = pd.concat(df_list)
df['pred'] = label_encoder.inverse_transform(df['pred'].values.reshape(-1, 1))
df[features] = feature_minmax.inverse_transform(df[features])

pred = np.array(df['pred'])
square = np.array(data['车道面积'])
DR=((100-pred)/15)**(1/0.412)
WA=(DR*square)/100
sum_DR=(sum(WA)/sum(square))*100
sum_PCI=100-15*(sum_DR**0.412)


print("好了")
print(sum_PCI)

df.to_csv("./data/predict_res_2023_pci.csv", index=False, encoding='utf_8_sig')

print("好了")


3）预测模型训练输入内容：
3.1）输入csv样例模版
训练模型：
训练资金模型：train5.csv 。训练pci模型：trainnew1.csv
新           ：                           ：trainnew2.csv

3.2）控制参数列表（名称、含义、数据类型、取值范围）
num_layers = 2 (网络层数, 越多时间越长，越精准，但过多容易过拟合，默认2.可选值1,3,4,5,6) (数据量大于3000时可以选择往上加)
window_size = 1 (预测间隔,默认为每一年,可选值2,3,4,5)
epochs = 200（训练回合数，越多时间越长，越精准，但过多容易过拟合，默认200.可选值150,250,300,350,400）

3.3）已有模型文件（输出）

预测资金：net_model_RNN3
预测pci：net_model_RNN6
模型权重的保存：net_model_RNN3_weight.pt
net_model_RNN6_weight.pt
     新加入数据后新的模型：net_model_RNNnew3
                           net_model_RNNnew6
在model文件夹内

4）PCI预测输入内容
4.1）输入csv样例（与资金预测相同，提供一个即可）
	预测资金：predict_true2023_2.csv
输出为：predict_res_2023_money.csv中的pred列，平均钱数为sum_money
	预测pci：predict_true2023_1.csv
输出为：predict_res_0717.csv",中的最后一列, 平均钱数为sum_PCI



