import numpy as np
import pickle
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

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
        std = c1.std(), c2.std(), c3.std\
            ()
        return mean,std


class Mnist(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file,header=None)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[[idx]].values
        label = sample[0][0]
        sample = np.delete(sample, 0).reshape(28, 28)
        sample = sample.astype(np.uint8)
        img = Image.fromarray(sample, mode='L')
        if self.transform:
            img = self.transform(img)

        # # check
        # plt.imshow(sample, cmap='Greys')
        # plt.show()

        return img, label
