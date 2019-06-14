import torch.nn as nn
import torch, random
from torch.utils.data import Dataset,DataLoader
import torch.nn.utils.rnn as rnn
from torch import optim
import matplotlib.pyplot as plt
from pprint import pprint


def auto_pad(samples_index,features_size):
    """
    先把所有seq转化为 0,1 的tensor后
    使用 pad_sequence
    需要给data 加padding
    """
    def t(x):
        tensor = torch.zeros(len(x),features_size)
        for i,v in enumerate(x):
            tensor[i][v] = 1
        return tensor

    tensor_list = [t(x) for x in samples_index]
    data_padding = rnn.pad_sequence(tensor_list, batch_first=True)

    data_len = [len(x) for x in samples_index]  # batch中,各seq的长度
    padded_sequence = rnn.pack_padded_sequence(data_padding, data_len, batch_first=True)

    return padded_sequence


class Net(nn.Module):
    def __init__(self,features_size,hidden_size):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(features_size, hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, features_size)

    def forward(self, x):
        output, (hn, hc) = self.lstm(x)

        raw_output, seq_len = rnn.pad_packed_sequence(output, batch_first=True)
        seqs = [seq[:length] for seq, length in zip(raw_output, seq_len)]
        output = torch.cat(seqs)

        output = self.output_layer(output)
        return output


with open('../data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
words = set(text)

word_to_index = {x: i for i, x in enumerate(words)}
index_to_word = {i: x for i, x in enumerate(words)}

raw_data = text.split("\n\n")
raw_data_index = [[word_to_index[i] for i in x] for x in raw_data]

labels_index = [x[1:] for x in raw_data_index]
samples_index = [x[:-1] for x in raw_data_index]

features_size = len(words)
BATCH_SIZE = 4
ITERATION = 200
seq_number = len(samples_index)

net = Net(features_size,50)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-4)


def batch_train(samples,labels):
    """
    :param samples: batch_size * seq_length
    :param lables: batch_size * seq_length
    :return:
    """
    optimizer.zero_grad()
    samples = auto_pad(samples,features_size)

    res = net(samples)
    # print(res.size())
    # print(labels.size())
    loss = criterion(res,labels)
    print(loss.item())
    loss.backward()
    optimizer.step()


def train():
    for epoch in range(ITERATION):
        n = 1
        while n*BATCH_SIZE <= seq_number:
            samples = samples_index[(n-1)*BATCH_SIZE:n*BATCH_SIZE]
            labels = labels_index[(n-1)*BATCH_SIZE:n*BATCH_SIZE]

            samples = sorted(samples, key=lambda x: len(x), reverse=True)
            labels = sorted(labels, key=lambda x: len(x), reverse=True)
            labels = [v for x in labels for v in x]
            labels = torch.tensor(labels)

            batch_train(samples,labels)
            n += 1
            # return


train()






