from a1_nn import SimpleNetwork
import numpy as np
import matplotlib.pyplot as plt


class MomentumNn(SimpleNetwork):
    def __init__(self, learning_rate, batch_size, input_node, output_node, layers, beta):
        super().__init__(learning_rate, batch_size, input_node, output_node, layers)
        self.beta = beta
        self.vdw = [np.zeros(x.shape) for x in self.w]
        self.vdb = [np.zeros(x.shape) for x in self.b]


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
        index = len(self.w) - i - 1
        if dz is None:
            dz = da * self.derivative_of_activation_function(z).T
        dw = np.dot(dz.T, ai_1.T)
        db = np.sum(dz.T, axis=1, keepdims=True)

        vdw = self.vdw[len(self.w)-i-1]
        vdb = self.vdb[len(self.w)-i-1]
        beta = self.beta
        vdw = beta * vdw + (1-beta)*dw
        vdb = beta * vdb + (1-beta)*db

        self.w[index] -= vdw * self.learning_rate
        self.b[index] -= vdb * self.learning_rate
        self.vdw[index] = vdw
        self.vdb[index] = vdb

        dai_1 = np.dot(dz, w)
        return dai_1


if __name__ == '__main__':
    nn = MomentumNn(0.001, 256, 784, 10, [300],0.9)
    loss = nn.train(100)
    plt.plot(range(1, len(loss)+1), loss, linewidth=0.5)
    plt.show()
    nn.accuracy()
