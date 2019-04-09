import torch.nn as nn
from dataset import Cifar10
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch


class ConvolutionalBlock(nn.Module):
    """
    block输入和输出,不一致
    """
    def __init__(self, in_channel, out_channel, stride):
        super(ConvolutionalBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.shotcurt = nn.Conv2d(in_channel, out_channel, 3, stride, 1)
        self.bn_shotcurt = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        residual = self.shotcurt(x)
        residual = self.bn_shotcurt(residual)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        output = x + residual
        return self.relu(output)


class IdentityBlock(nn.Module):
    """
    block输入和输出,一致
    """
    def __init__(self, in_channel, out_channel):
        super(IdentityBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        output = x + residual
        return self.relu(output)


class SimpleResNet(nn.Module):
    """
    简化resnet-18 用于 cifar-10( image resize to 3*38*38 )
    """
    def __init__(self):
        super(SimpleResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(3, 1)

        self.a1 = IdentityBlock(16, 16)
        self.a2 = IdentityBlock(16, 16)

        self.b1 = ConvolutionalBlock(16, 32, 2)
        self.b2 = IdentityBlock(32, 32)

        self.c1 = ConvolutionalBlock(32, 64, 2)
        self.c2 = IdentityBlock(64, 64)

        self.d1 = ConvolutionalBlock(64, 128, 2)
        self.d2 = IdentityBlock(128, 128)

        self.pool2 = nn.AvgPool2d(4)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.a1(x)
        x = self.a2(x)

        x = self.b1(x)
        x = self.b2(x)

        x = self.c1(x)
        x = self.c2(x)

        x = self.d1(x)
        x = self.d2(x)

        x = self.pool2(x)

        x = x.view(-1,128)
        x = self.fc(x)
        return x


temp = Cifar10()
mean, std = temp.mean_std()
print(mean)
print(std)


transform = transforms.Compose([
    transforms.Resize((38, 38)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

train_set = Cifar10('train', transform)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

test_set = Cifar10('test', transform)
test_loader = DataLoader(test_set, batch_size=256)

# device = torch.device('cuda:0' if torch.cuda.is_available() else ' cpu')
# print(device)
resnet = SimpleResNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet.parameters(), lr=1e-3)


def train(iteration):
    loss_list = []
    for epoch in range(1, iteration):
        epoch_loss = 0
        n = 0
        for i in train_loader:
            optimizer.zero_grad()
            samples = i[0]
            labels = i[1].long()

            res = resnet(samples)
            loss = criterion(res, labels)
            loss.backward()
            optimizer.step()
            print(loss.item())
            epoch_loss += loss.item()
            n += 1
        loss_list.append(epoch_loss/n)
    plt.plot(range(1, len(loss_list) + 1), loss_list, linewidth=0.5)
    plt.show()


def test():
    correct = 0
    for data in test_loader:
        res = resnet(data[0])
        label = data[1]
        correct += label.eq(res.max(1)[1]).sum()
    print(correct)
    print(f"{'%.3f' % (correct.item()/100)}%")


train(100)
test()