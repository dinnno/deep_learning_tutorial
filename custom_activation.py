import numpy as np
import pandas as pd
from collections import OrderedDict

import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.autograd import Function
from torch.nn.parameter import Parameter

import torch.optim as optim
import torch.nn.functional as F

from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor()])

tr_data = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
tr_loader = torch.util.data.DataLoader(tr_data, batch_size=64, shuffle=True)

def trainer(model, tr_loader):

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    learning_rate = 0.0001
    epochs = 5

    for epoch in range(epochs):
        running_loss = 0
        for images, labels in tr_loader:
            images = images.view(images.shape[0], -1)
            output = model(images)
            loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
