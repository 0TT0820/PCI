import pandas as pd
import torch
import numpy as np
from entity.dataloader import get_dataloader
import joblib

test_path = "./data/yuanben2022.csv"
data = pd.read_csv(test_path)

pred = np.array(data['当年pci'])
square = np.array(data['车道面积'])
DR=((100-pred)/15)**(1/0.412)
WA=(DR*square)/100
sum_DR=(sum(WA)/sum(square))*100
sum_PCI=100-15*(sum_DR**0.412)


print("好了")
print(sum_PCI)