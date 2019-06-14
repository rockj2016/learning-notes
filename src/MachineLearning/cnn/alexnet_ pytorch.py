import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn as nn
from torch import optim
from dataset import Cifar10


class AlexNet(nn.Module):
    """
    简化的 AlexNet 用于32*32 cifar10
    """
    def __init__(self):
        super(AlexNet, self).__init__()
        """
        input : 3 * 10 * 32
        output : 20 * 14 * 14
        """
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 20, 5),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.LocalResponseNorm(5)
        )
        """
        input : 20 * 14 * 14
        output : 40 * 6 * 6
        """
        self.conv2 = nn.Sequential(
            nn.Conv2d(20, 40, 3),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.LocalResponseNorm(5)
        )
        """
        input : 40 * 6 * 6
        output : 60 * 6 * 6
        """
        self.conv3 = nn.Sequential(
            nn.Conv2d(40, 60, 3, padding=1),
            nn.ReLU(),
        )
        """
        input : 60 * 6 * 6
        output : 60 * 6 * 6
        """
        self.conv4 = nn.Sequential(
            nn.Conv2d(60, 60, 3, padding=1),
            nn.ReLU(),

        )
        """
        input : 60 * 6 * 6
        output : 40 * 3 * 3
        """
        self.conv5 = nn.Sequential(
            nn.Conv2d(60, 40, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        """
        input : m * 360
        output : m * 360
        """
        self.fc6 = nn.Sequential(
            nn.Linear(360, 360),
            nn.ReLU(),
            nn.Dropout(0.8)
        )
        """
        input : m * 864
        output : m *864
        """
        self.fc7 = nn.Sequential(
            nn.Linear(360, 360),
            nn.ReLU(),
            nn.Dropout(0.8)
        )
        """
        input : m * 864
        output : m * 10
        """
        self.fc8 = nn.Sequential(
            nn.Linear(360, 10),
        )
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(-1,360)
        x = self.fc6(x)
        x = self.fc7(x)
        x = self.fc8(x)
        return x


temp = Cifar10()
mean, std = temp.mean_std()
print(mean)
print(std)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

train_set = Cifar10('train', transform)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

test_set = Cifar10('test', transform)
test_loader = DataLoader(test_set, batch_size=256)


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        import math
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))


net = AlexNet()
net.apply(weight_init)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=5e-4)


def train(n):
    loss_list = []
    for epoch in range(1, n+1):
        running_loss = 0
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            samples = data[0]
            labels = data[1].long()
            print(samples.size())
            res = net(samples)
            print(res.size())
            print(labels.size())
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


train(100)
test()


