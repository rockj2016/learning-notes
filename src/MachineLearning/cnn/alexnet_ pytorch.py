import pickle
import numpy as np
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn as nn


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


class AlexNet(nn.Module):
    def __init__(self):
        pass


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0], std=[1]),
])

train_set = Cifar10('train', transform)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

test_set = Cifar10('test', transform)
test_loader = DataLoader(test_set, batch_size=256)

for i in train_set:
    print(i[0])
    break





