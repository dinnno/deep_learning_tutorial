'''
how to use BIC data (e.g., ImageNet)

open dataset is really bic, so we can use minibatch gradient descent


torch.utils.data.Dataset 클래스를 상속받아 사용할 수 있음

__len__() : 데이터 셋의 총 데이터 갯수

__getitem__() : 어떠한 idx를 받았을 때, 그에 상응하는 입출력 데이터를 반환

custom data를 만들었다면, torch.utils.data.DataLoader 클래스를 사용하여,
DataLoader를 쉽게 만들 수 있음

enumerate(dataloader) : minibatch idx, data 받음
len(dataloader) : 한 epoch minibatch 갯수
'''

import torch
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class CustomDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])

        return x, y

cus_data = cus_data

cus_dataloader = DataLoader(cus_data, batch_size=16, shuffle=True)

