import torch
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import os

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from PIL import Image

from glob import glob

LABEL_DICT = { 'car' : 0, 'airplane': 1}

class CustomData(Dataset):
    def __init__(self, base_root, mode='train'):
        # data_path_setting
        self.base_root = './data/sung/Cifar10'
        self.mode = mode
        self.width = width
        self.height = height

        if self.mode == 'train':
            self.data_path = os.path.join(base_root, "train")

        elif self.mode == 'val':
            self.data_path = os.path.join(base_root, 'val')

        elif self.mode == 'test':
            self.data_path = os.path.join(base_root, 'test')

        self.img = glob(data_path + '/*/*.jpg')    

        # data_preprocesssing 

        if self.mode == "train":
            self.transform = transforms.Compose(transforms.RandomCrop(32, padding=4),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.RandomRotation(15),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
        else:
            self.transform = transforms.Compose(transforms.ToTensor(),
                                                transforms.Normalize([], []))
        
    def __getitem__(self, idx):

        img_path = self.img[idx]
        
        target = LABEL_DICT[img_path.split('/')[-2]]

        target = torch.tensor(target).long()

        img = Image.open(img_path).conver('RGB')

        img = torch.from_numpy(np.array(img))         
        
        if self.transform is not None:
            img = self.transform(img)

        return img, target
        

    def __len(self):
        return len(self.data_list)

batch_size = 4

tr_set = CustomData('train', transform)
val_set = CustomData('val', transform)
test_set = CustomData('test', transform)

tr_loader = DataLoader(tr_set, batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5) #in_channels, out_channels, kernel
        self.pool = nn.Maxpool2d(2, 2) # kernel_size, stride
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.Maxpool2d(2, 2)
        
        self.fc1 = nn.Linear(16*5*5, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

device = 'cuda'

net = Net()
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters, lr=0.001)

num_epochs = 100

for epoch in range(num_epochs):

    net.train()
    running_loss = 0

    for i, data in enumerate(tr_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = Net(inputs)
        loss = criterion(outpus, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, pred = torch.max(outputs, 1)
    
    running_loss /= len(tr_loader)

    if epoch % 10 == 0:
        net.eval()
        correct = 0
        total = 0
        for j, data in enumerate(val_loader, 0):
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                outputs = net(images)
                _, pred = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()

        acc = 100 * (correct / total)
