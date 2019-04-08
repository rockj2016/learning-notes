import pickle
import numpy as np
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn as nn
from torch import optim
import torch


# dataset download from https://www.cs.toronto.edu/~kriz/cifar.html
def load_file(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data


def get_train_set():
    labels = None
    data = None
    for i in range(1, 6):
        path = f'../data/cifar10/data_batch_{i}'
        temp = load_file(path)
        if labels is not None:
            labels = np.concatenate((labels, temp['labels']), axis=0)
        else:
            labels = temp['labels']
        if data is not None:
            data = np.concatenate((data, temp['data']), axis=0)
        else:
            data = temp['data']
    return data, labels


def get_test_set():
    path = f'../data/cifar10/test_batch'
    temp = load_file(path)
    labels = temp['labels']
    data = temp['data']
    return data,labels


class Cifar10(Dataset):
    def __init__(self, mode='train', transform=None):
        if mode == 'train':
            self.data,self.labels = get_train_set()
        else:
            self.data,self.labels = get_test_set()
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx].reshape(3,32,32)
        sample = np.transpose(sample, (1, 2, 0))
        label = self.labels[idx]
        img = Image.fromarray(sample, mode='RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

    def mean_std(self):
        c1 = []
        c2 = []
        c3 = []
        for i in self.data:
            data = i.reshape(3,1024)
            c1.append(data[0])
            c2.append(data[1])
            c3.append(data[2])
        c1 = np.array(c1)/255
        c2 = np.array(c2)/255
        c3 = np.array(c3)/255
        mean = c1.mean(), c2.mean(), c3.mean()
        std = c1.std(), c2.std(), c3.std()
        return mean,std


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
optimizer = optim.Adam(net.parameters(), lr=1e-4)


def train(n):
    loss_list = []
    for epoch in range(1, n+1):
        running_loss = 0
        for i,data in enumerate(train_loader):
            optimizer.zero_grad()
            samples = data[0]

            labels = data[1].long()
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


train(100)
test()


