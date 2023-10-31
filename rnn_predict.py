# -- coding: utf-8 --
"""
Time: 
    2022/8/15 23:00
Author:
    tong
Reference:
    
"""
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
test_path = "./data/predict2024.csv"
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

df.to_csv("./data/predict_res_2024_pci.csv", index=False, encoding='utf_8_sig')

print("好了")