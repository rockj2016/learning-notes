import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from dataset import Mnist

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1307], std=[0.3081]),
])

train_set = Mnist('../../data/mnist_train.csv', transform)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

test_set = Mnist('../../data/mnist_test.csv', transform)
test_loader = DataLoader(test_set, batch_size=256)


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 120, 5)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))

        x = x.view(-1, 120)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


criterion = nn.CrossEntropyLoss()
net = LeNet()
optimizer = optim.Adam(net.parameters(), lr=1e-3)


def train(n):
    loss_list = []
    for epoch in range(1, n+1):
        running_loss = 0
        for i,data in enumerate(train_loader):
            optimizer.zero_grad()
            samples = data[0]
            labels = data[1]
            res = net(samples)
            loss = criterion(res, labels)
            # print('%.3f' % loss.item())

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:  # print every 10 mini-batches
                print(f"epoch : {epoch}  "
                      f"batch : {i-9}-{i}  "
                      f"loss : {'%.3f' % (running_loss/10)}")
                loss_list.append('%.3f' % (running_loss/10))
                running_loss = 0.0

    loss_list = [float(x) for x in loss_list]
    plt.plot(range(1, len(loss_list) + 1), loss_list, linewidth=0.5)
    plt.show()


def test():
    correct = 0
    for data in test_loader:
        res = net(data[0])
        label = data[1]
        correct += label.eq(res.max(1)[1]).sum()
    print(correct)
    print(f"{'%.3f' % (correct.item()/100)}%")


train(10)
test()