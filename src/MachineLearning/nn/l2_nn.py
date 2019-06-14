from .nn import SimpleNetwork
import numpy as np
import matplotlib.pyplot as plt


class L2Nn(SimpleNetwork):
    def __init__(self,learning_rate, batch_size, input_node, output_node, layers, lambd):
        super().__init__(learning_rate, batch_size, input_node, output_node, layers)
        self.lamdb = lambd

    def backward(self, i, n, w, da, z, ai_1, dz=None):
        """
        :param i: 第几层
        :param n: 第几轮迭代
        :param w: wi 第 i 层 w
        :param da: dai 第 i 层 a
        :param z: di 第 i 层 z
        :param ai_1: ai-1 第 i-1 层 a
        :param dz: dzi 如传入dzi则函数内不再根据 da,z计算dzi
        :return: dai-1 用于前一层求导计算
        """
        index = len(self.w)-i-1
        if dz is None:
            dz = da * self.derivative_of_activation_function(z).T
        m = dz.shape[1]
        dw = np.dot(dz.T, ai_1.T) + self.lamdb / (2 * m) * self.w[index]
        db = np.sum(dz.T, axis=1, keepdims=True)
        self.w[index] -= dw * self.learning_rate
        self.b[index] -= db * self.learning_rate

        dai_1 = np.dot(dz, w)
        return dai_1

    def calculate_loss(self):
        x = np.concatenate(self.x, axis=1)
        y = np.concatenate(self.y, axis=1)
        m = x.shape[1]

        length = len(self.w) - 1
        for i, (w, b) in enumerate(zip(self.w, self.b)):
            if i == length:
                temp = self.forward(x, w, b, self.softmax)
            else:
                temp = self.forward(x, w, b, self.activation_function)
            x = temp[2]
        p = x
        loss = self.cross_entropy(p, y)
        reg = self.lamdb / (2 * m) * sum(np.linalg.norm(x)**2 for x in nn.w)
        print(f'loss:{loss+reg}')
        return loss + reg


if __name__ == '__main__':
    nn = L2Nn(0.001, 256, 784, 10, [300], 0.1)
    loss = nn.train(100)
    plt.plot(range(1, len(loss)+1), loss, linewidth=0.5)
    plt.show()
    nn.accuracy()
