import torch.nn as nn
import torch, random
from torch.utils.data import Dataset,DataLoader
from torch import optim
import matplotlib.pyplot as plt


def get_data():
    with open('../data/input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
        words = set(text)
        word_to_index = {x: i for i, x in enumerate(words)}
        index_to_word = {i: x for i, x in enumerate(words)}

        data = text.split("\n\n")
        data = [[word_to_index[i] for i in x] for x in data]
        labels = [x[1:] for x in data]
        data = [x[:-1] for x in data]

        return data, labels, index_to_word, word_to_index


class TextS(Dataset):
    def __init__(self):
        self.data, self.lables, self.index_to_word, self.word_to_index \
            = get_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        samples = self.data[idx]
        labels = self.lables[idx]
        labels = torch.tensor(labels)
        sample_tensor = torch.zeros((len(samples),1,65))
        for i,v in enumerate(samples):
            sample_tensor[i][0][v] = 1
        return sample_tensor, labels


class RnnGru(nn.Module):
    def __init__(self, x_size, h_size):
        super(RnnGru, self).__init__()
        self.x_size = x_size
        self.h_size = h_size
        self.zt = nn.Linear((self.x_size+self.h_size), self.h_size)
        self.rt = nn.Linear((self.x_size+self.h_size), self.h_size)
        self.ht_hat = nn.Linear((self.x_size+self.h_size), self.h_size)
        self.yp = nn.Linear(self.h_size, self.x_size)
        self.tanh = nn.Tanh()
        self.sigmod = nn.Sigmoid()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x, h):
        i = torch.cat((x, h),1)
        zt = self.sigmod(self.zt(i))
        rt = self.sigmod(self.rt(i))
        i_2 = torch.cat((x, rt * h),1)
        ht_hat = self.tanh(self.ht_hat(i_2))
        h = (1 - zt) * h + zt * ht_hat
        y = self.yp(h)
        output = self.logsoftmax(y)
        return h,output


net = RnnGru(65,200)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=5e-4)

texts = TextS()

train_loader = DataLoader(texts, batch_size=1, shuffle=True)


def train(iteration):
    loss_list =[]
    for i in range(iteration):
        for datas, labels in train_loader:
            loss = 0
            optimizer.zero_grad()
            h_prev = torch.zeros((1, 200))
            for x, y in zip(datas[0], labels[0]):
                h_prev,res = net(x, h_prev)
                loss += criterion(res, y.reshape(1))

            loss.backward()
            optimizer.step()

            print(loss.item()/len(datas[0]))
            loss_list.append(loss.item()/len(datas[0]))

        print(sample(100))
    plt.plot(range(1, len(loss_list) + 1), loss_list, linewidth=0.5)
    plt.show()




train(10)
sample(100)
