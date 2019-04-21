from nn import SimpleNetwork
import numpy as np
import matplotlib.pyplot as plt


class AdamNn(SimpleNetwork):
    def __init__(self, learning_rate, batch_size, input_node, output_node, layers):
        super().__init__(learning_rate, batch_size, input_node, output_node, layers)

        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.esp = 10e-8

        self.vdw = [np.zeros(x.shape) for x in self.w]
        self.vdb = [np.zeros(x.shape) for x in self.b]
        self.sdw = [np.zeros(x.shape) for x in self.w]
        self.sdb = [np.zeros(x.shape) for x in self.b]

    def backward(self, i, n, w, da, z, ai_1, dz=None):
        """
        :param i: 第几层
        :param n: batch迭代次数,每个处理一个batch+1
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

        vdw = self.vdw[index]
        vdb = self.vdb[index]
        beta_1 = self.beta_1
        vdw = beta_1 * vdw + (1 - beta_1) * dw
        vdb = beta_1 * vdb + (1 - beta_1) * db

        sdw = self.sdw[index]
        sdb = self.sdb[index]
        beta_2 = self.beta_2
        sdw = beta_2 * sdw + (1 - beta_2) * dw ** 2
        sdb = beta_2 * sdb + (1 - beta_2) * db ** 2

        sdw_ = sdw / (1 + beta_2 ** n)
        sdb_ = sdb / (1 + beta_2 ** n)
        vdw_ = vdw / (1 + beta_1 ** n)
        vdb_ = vdb / (1 + beta_1 ** n)

        self.w[index] -= self.learning_rate * vdw_ / (np.sqrt(sdw_) + self.esp)
        self.b[index] -= self.learning_rate * vdb_ / (np.sqrt(sdb_) + self.esp)
        self.sdw[index] = sdw
        self.sdb[index] = sdb
        self.vdw[index] = vdw
        self.vdb[index] = vdb

        dai_1 = np.dot(dz, w)
        return dai_1


if __name__ == '__main__':
    nn = AdamNn(0.001, 256, 784, 10, [300])
    loss = nn.train(10)
    plt.plot(range(1, len(loss)+1), loss, linewidth=0.5)
    plt.show()
    nn.accuracy()


