from a1_nn import SimpleNetwork
import numpy as np
import matplotlib.pyplot as plt


class DropOutNn(SimpleNetwork):
    def __init__(self, learning_rate, batch_size, input_node, output_node, layers, keep_prob):
        super().__init__(learning_rate, batch_size, input_node, output_node, layers)
        self.keep_prob = keep_prob

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
        mask = None
        if activation_function == self.activation_function:
            # print(np.random.rand(ai.shape[0], ai.shape[1]))
            dp = np.random.rand(ai.shape[0], ai.shape[1]) <= self.keep_prob
            mask = dp / self.keep_prob
            ai = ai * mask

        return ai_1, z, ai, w, mask

    def backward(self, i, n, w, da, z, ai_1, mask, dz=None):
        """
        :param i: 第几层
        :param n: 第几轮迭代
        :param w: wi 第 i 层 w
        :param da: dai 第 i 层 da
        :param z: di 第 i 层 z
        :param ai_1: ai-1 第 i-1 层 a
        :param dz: dzi 如传入dzi则函数内不再根据da,z计算dzi
        :param mask: dropout mask
        :return: dai-1 用于前一层求导计算
        """
        if dz is None:
            dz = da * self.derivative_of_activation_function(z).T
            dz = dz * mask.T
        dw = np.dot(dz.T, ai_1.T)
        db = np.sum(dz.T, axis=1, keepdims=True)
        self.w[len(self.w)-i-1] -= dw * self.learning_rate
        self.b[len(self.b)-i-1] -= db * self.learning_rate

        dai_1 = np.dot(dz, w)
        return dai_1

    def train(self, iteration):
        # 迭代次数
        loss = []
        for n in range(1, iteration+1):
            # 遍历所有 batches
            for x, y in zip(self.x, self.y):
                # 遍历所有 layers
                # forward
                batch_size = x.shape[1]
                forward_data = []
                length = len(self.w)-1
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
                    ai_1, z, ai, w, mask = v
                    da = self.backward(i, n, w, da, z, ai_1, mask, dz)
                    dz = None
            loss.append(self.calculate_loss())
        return loss


if __name__ == '__main__':
    nn = DropOutNn(0.01, 256, 784, 10, [300], 0.8)
    loss = nn.train(100)
    nn.accuracy()
    plt.plot(range(1, len(loss)+1), loss, linewidth=0.5)
    plt.show()

