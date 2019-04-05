import numpy as np
from functools import partial
import matplotlib.pyplot as plt


def get_data():
    with open('../data/mnist_train.csv', 'r') as f:
        data = [x.strip().split(',') for x in f]
        data = np.asarray(data, dtype='float').T
        # print(data[0][:10])
        y = np.zeros((10, len(data[0])))
        for i, x in enumerate(data[0]):
            y[:, i][int(x)] = 1
        x = np.delete(data, 0, 0)
        x = (x / 255.0 * 0.99) + 0.01
        x = x - x.mean(axis=1, keepdims=True)
        x = x / np.sqrt(x.var(axis=1, keepdims=True) + 10e-8)
        return x, y


def get_test_data():
    with open('../data/mnist_test.csv', 'r') as f:
        data = [x.strip().split(',') for x in f]
        data = np.asarray(data, dtype='float').T
        # print(data[0][:10])
        y = data[0]
        x = np.delete(data, 0, 0)
        x = ((x / 255.0) * 0.99) + 0.01
        x = x - x.mean(axis=1, keepdims=True)
        x = x / np.sqrt(x.var(axis=1, keepdims=True) + 10e-8)
        return x, y


class SimpleNetwork:
    def __init__(self, learning_rate, batch_size, input_node, output_node, layers):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.layers = layers

        self.x, self.y = self.get_batch_data()
        self.input_node = input_node
        self.output_node = output_node

        self.activation_function = self.relu
        self.derivative_of_activation_function = partial(self.relu, derivative=True)

        self.w, self.b = self.get_w_b()

    @staticmethod
    def softmax(x):
        if x.shape[1] == 1:
            exp = np.exp(x - np.max(x))
            return exp / np.sum(exp)
        else:
            exp = np.exp(x - np.max(x, axis=0, keepdims=True))
            return exp / np.sum(exp, axis=0, keepdims=True)

    @staticmethod
    def relu(x, derivative=False):
        if derivative:
            x[x <= 0] = 0
            x[x > 0] = 1
            return x
        return np.maximum(x, 0)

    @staticmethod
    def cross_entropy(p, y, epsilon=1e-12):
        p = np.clip(p, epsilon, 1. - epsilon)
        m = p.shape[1]
        ce = -np.sum(y * np.log(p)) / m
        return ce

    def get_w_b(self):
        w_list = []
        b_list = []
        for index, layer in enumerate(self.layers):
            if index == 0:
                w = np.random.randn(layer, self.input_node) * np.sqrt(2 / self.input_node)
            else:
                w = np.random.randn(layer, self.layers[index-1]) * np.sqrt(2 / self.layers[index-1])
            b = np.zeros((layer, 1))
            w_list.append(w)
            b_list.append(b)

        w = np.random.randn(self.output_node, self.layers[-1]) * np.sqrt(2 / self.layers[-1])
        b = np.zeros((self.output_node, 1))
        w_list.append(w)
        b_list.append(b)
        return w_list,b_list

    def get_batch_data(self):
        x, y = get_data()
        m = x.shape[1]
        batch_split_index = []
        index = self.batch_size
        while index <= m:
            batch_split_index.append(index)
            index += self.batch_size
        x_batches = np.split(x, batch_split_index, axis=1)
        y_batches = np.split(y, batch_split_index, axis=1)

        return x_batches,y_batches

    def forward(self, ai_1, w, b, activation_function):
        """
        :param ai_1: ai-1
        :param w: wi
        :param b: bi
        :param activation_function: activation_function
        :return: ai-1, zi, ai, wi, bi
        """
        z = np.dot(w, ai_1) + b
        ai = activation_function(z)
        return ai_1, z, ai, w

    def backward(self, i, n, w, da, z, ai_1, dz=None):
        """
        :param i: 第几层
        :param n: 第几轮迭代
        :param w: wi 第 i 层 w
        :param da: dai 第 i 层 a
        :param z: di 第 i 层 z
        :param ai_1: ai-1 第 i-1 层 a
        :param dz: dzi 如传入dzi则函数内不再根据da,z计算dzi
        :return: dai-1 用于前一层求导计算
        """

        if dz is None:
            dz = da * self.derivative_of_activation_function(z).T
        dw = np.dot(dz.T, ai_1.T)
        db = np.sum(dz.T, axis=1, keepdims=True)
        self.w[len(self.w)-i-1] -= dw * self.learning_rate
        self.b[len(self.b)-i-1] -= db * self.learning_rate

        dai_1 = np.dot(dz, w)
        return dai_1

    def calculate_loss(self):
        x = np.concatenate(self.x, axis=1)
        y = np.concatenate(self.y, axis=1)

        length = len(self.w) - 1
        for i, (w, b) in enumerate(zip(self.w, self.b)):
            if i == length:
                temp = self.forward(x, w, b, self.softmax)
            else:
                temp = self.forward(x, w, b, self.activation_function)
            x = temp[2]
        p = x
        loss = self.cross_entropy(p, y)
        print(f'loss:{loss}')
        return loss

    def train(self, iteration):
        # 迭代次数
        loss = []
        n = 1
        for i in range(iteration):
            # 遍历所有 batches
            for x, y in zip(self.x, self.y):
                # 遍历所有 layers
                # forward
                batch_size = x.shape[1]
                forward_data = []
                length = len(self.w) - 1
                for i, (w, b) in enumerate(zip(self.w, self.b)):
                    if i == length:
                        temp = self.forward(x, w, b, self.softmax)
                    else:
                        temp = self.forward(x, w, b, self.activation_function)
                    forward_data.append(temp)
                    x = temp[2]
                p = x

                # backward
                dz = (p - y).T / batch_size
                da = None
                forward_data.reverse()
                for i, v in enumerate(forward_data):
                    ai_1, z, ai, w = v
                    da = self.backward(i, n, w, da, z, ai_1, dz)
                    dz = None
                n += 1
            loss.append(self.calculate_loss())
        return loss

    def predict(self, features):
        features = features.reshape((self.input_node, 1))

        a = features
        for i in range(len(self.w)):
            w = self.w[i]
            b = self.b[i]
            if i+1 == len(self.w):
                temp = self.forward(a, w, b, self.softmax)
            else:
                temp = self.forward(a, w, b, self.activation_function)
            a = temp[2]

        a = a.tolist()
        return a.index(max(a))

    def accuracy(self):
        false = 0
        x, y = get_test_data()
        for i in zip(x.T, y):
            predict_number = self.predict(i[0])
            if predict_number != i[1]:
                false += 1
        print('right_ratio:', (len(x.T)-false)/100)
        return (len(x.T)-false)/10000


if __name__ == '__main__':
    nn = SimpleNetwork(0.01, 256, 784, 10, [300])
    loss = nn.train(100)
    plt.plot(range(1, len(loss)+1), loss, linewidth=0.5)
    plt.show()
    nn.accuracy()

