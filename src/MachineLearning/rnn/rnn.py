import numpy as np
import math, copy
import matplotlib.pyplot as plt


def softmax(x):
    if x.shape[1] == 1:
        exp = np.exp(x - np.max(x))
        return exp / np.sum(exp)
    else:
        exp = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp / np.sum(exp, axis=0, keepdims=True)


class NetWork:
    def __init__(self, hidden_size, truncation_size):
        self.hidden_size = hidden_size
        self.truncation_size = truncation_size
        self.data = self.get_data()
        self.features = set(self.data)
        self.feature_size = len(self.features)
        self.feature_index_dict = {x: i for i, x in enumerate(self.features)}
        self.index_feature_dict = {i: x for i, x in enumerate(self.features)}
        self.parameters = self.init_parameters()
        self.ds = {k: np.zeros_like(v) for k, v in self.parameters.items()}
        self.dv = {k: np.zeros_like(v) for k, v in self.parameters.items()}

        print(self.feature_size)
        print(len(self.data))

    def get_data(self):
        # with open('../data/text.txt', 'r', encoding='utf-8') as f:
        with open('../data/input.txt', 'r', encoding='utf-8') as f:
            data = f.read()
            return data
            # return data[0:int(len(data)/150)]

    def init_parameters(self):
        wxh = np.random.randn(self.hidden_size,self.feature_size) * 0.001
        whh = np.random.randn(self.hidden_size,self.hidden_size) * 0.001
        bh = np.zeros((self.hidden_size,1))
        why = np.random.randn(self.feature_size,self.hidden_size) * 0.001
        by = np.zeros((self.feature_size,1))
        return {
            'wxh': wxh,
            'whh': whh,
            'bh': bh,
            'why': why,
            'by': by,
        }

    def train(self, iteration):
        loss_list = []
        smooth_loss = -np.log(1.0 / self.feature_size) * self.truncation_size

        times = 1
        for i in range(iteration):
            i = 0
            n = math.floor(len(self.data)/self.truncation_size)
            h_prev = np.zeros((self.hidden_size,1))
            while i < n:
                input_data = self.data[i:i+self.truncation_size]
                input_data = [self.feature_index_dict[x] for x in input_data]
                label_data = self.data[i+1:i+self.truncation_size+1]
                label_data = [self.feature_index_dict[x] for x in label_data]
                labels, hs, xs, ys, ps, h_prev, loss = self.forward(input_data, h_prev, label_data)
                dwxh, dwhh, dbh, dwhy, dby = self.backword(labels, hs, xs, ps)
                d_dict = {
                    'wxh': dwxh,
                    'whh': dwhh,
                    'bh': dbh,
                    'why': dwhy,
                    'by': dby,
                }
                self.update_parameters(d_dict, times)
                times += 1

                smooth_loss = smooth_loss * 0.999 + loss * 0.001

                i += self.truncation_size

            loss_list.append(smooth_loss[0])
            print(smooth_loss)
            print(self.sample(1,200))
            print('--'*20)
            print('\n')

        plt.plot(range(1, len(loss_list) + 1), loss_list, linewidth=0.5)
        plt.show()

    def update_parameters(self, d, n):

        beta_1 = 0.9
        beta_2 = 0.999
        learning_rate = 1e-3
        esp = 10e-8

        # for k, v in d.items():
        #     self.parameters[k] -= v * learning_rate

        for k, v in d.items():
            vd = self.dv[k]
            sd = self.ds[k]

            vd = beta_1 * vd + (1 - beta_1) * v
            sd = beta_2 * sd + (1 - beta_2) * (v ** 2)

            vd_ = vd / (1 - beta_1 ** n)
            sd_ = sd / (1 - beta_2 ** n)

            self.parameters[k] -= learning_rate * vd_ / (np.sqrt(sd_) + esp)
            self.ds[k] = sd
            self.dv[k] = vd

    def forward(self, data, h_prev, labels):
        hs = []
        xs = []
        ys = []
        ps = []
        loss = 0
        h_input = copy.copy(h_prev)
        for index in range(len(data)):
            x = np.zeros((self.feature_size, 1))
            x[data[index]] = 1
            h = np.tanh(
                    np.dot(self.parameters['whh'],h_prev) + np.dot(self.parameters['wxh'],x) + self.parameters['bh']
            )
            y = np.dot(self.parameters['why'], h) + self.parameters['by']
            p = softmax(y)

            loss += -np.log(p[labels[index]])
            # loss += -np.log(np.maximum(p[labels[index]],float(10e-10)))

            # cache
            hs.append(h)
            xs.append(x)
            ys.append(y)
            ps.append(p)
            h_prev = h
        hs.append(h_input)

        return labels, hs, xs, ys, ps, h_prev, loss

    def backword(self, labels, hs, xs, ps):
        dwxh = np.zeros_like(self.parameters['wxh'])
        dwhh = np.zeros_like(self.parameters['whh'])
        dbh = np.zeros_like(self.parameters['bh'])
        dwhy = np.zeros_like(self.parameters['why'])
        dby = np.zeros_like(self.parameters['by'])
        dhnext = np.zeros_like(hs[0])
        for index in reversed(range(len(xs))):
            # print(ps[index],labels[index])
            ps[index][labels[index]] = ps[index][labels[index]] - 1
            dy = ps[index]
            dwhy += np.dot(dy, hs[index].T)
            dby += dy

            dh = np.dot(self.parameters['why'].T, dy) + dhnext
            dh_raw = (1 - hs[index] * hs[index]) * dh
            dbh += dh_raw

            dwxh += np.dot(dh_raw, xs[index].T)

            dwhh += np.dot(dh_raw, hs[index - 1].T)

            dhnext = np.dot(self.parameters['whh'].T, dh_raw)

        np.clip(dwxh, -5, 5, out=dwxh)
        np.clip(dwhh, -5, 5, out=dwhh)
        np.clip(dbh, -5, 5, out=dbh)
        np.clip(dwhy, -5, 5, out=dwhy)
        np.clip(dby, -5, 5, out=dby)

        return dwxh, dwhh, dbh, dwhy, dby

    def sample(self, start_index, n):
        x = np.zeros((self.feature_size, 1))
        h = np.zeros((self.hidden_size, 1))
        x[start_index] = 1
        xs = []
        for i in range(n):
            h = np.tanh(
                np.dot(self.parameters['whh'], h) + np.dot(self.parameters['wxh'], x) + self.parameters['bh']
            )
            y = np.dot(self.parameters['why'], h) + self.parameters['by']
            p = softmax(y)
            # ix = np.argmax(p)
            ix = np.random.choice(range(self.feature_size), p=p.ravel())
            x = np.zeros((self.feature_size, 1))
            x[ix] = 1
            xs.append(ix)
        return ''.join([self.index_feature_dict[x] for x in xs])


n = NetWork(100, 32)
n.train(1000)
