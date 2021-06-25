import torchvision
import torchvision.transforms as transforms

from torch.utils.data.dataloader import DataLoader

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

tr_transform = transforms.Compose([transforms.ToTensor(),
                                transforms.normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

val_transform = transforms.Compose([transforms.ToTensor(),
                                transforms.normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


batch_size = 4

classes = ['Car', 'Truck', '2,', '3,', '4', '5', '6', '7', '8', '9']

tr_set = torch.dataset.CIFAR10(path='./data/sung/dataset/cifar10', train=True, download=True, transform=tr_transform)
tr_loader = DataLoader(tr_set, batch_size, shuffle=True)

val_set = torch.dataset.CIFAR10(path='./data/sung/dataset/cifar10', train=False, download=True, transform=val_transform)
val_loader = DataLoader(val_set, batch_size, shuffle=False)


class Net(nn.module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) #conv2d input : in_channels, out_channels, kernel
        self.pool = nn.Maxpool2d(2, 2)

        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.Maxpool2d(2, 2)

        self.fc1 = nn.Linear(16*5*5, 64)
        self.fc2 = nn.Linear(64 ,32)
        self.fc3 = nn.Linear(32, 10) # 10 : num_class of cifar10

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # flatten

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

device = 'cuda'

net = Net()
net = net.to(device)

criterion = nn.CrossEntropyLoss() # 어디 모듈이었는지 까먹음'
optimizer = optim.Adam(net.parameters(), lr=0.001)

epoch_num = 100

for epoch in range(epoch_num):
    ## training
    correct = 0
    total = 0

    net.train()
    running_loss = 0.0
    for data in tr_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = net(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, pred = torch.max(outputs)

    running_loss /= len(tr_loader)


    if epoch % 10 == 0:
        net.eval()
        for i, data in enumerate(val_loader, 0):
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                outputs = net(images)