import numpy as np


def conv_forward(input_data, filters, bias, padding=0, stride=1):
    """
    :param input_data: input  m x nc x iw x ih
    :param filters: filter nf x nc x fw x fh  n为filter数量
    :param bias: bias 每一filter对应一个实数b nf x 1
    :param padding: padding layer:l
    :param stride: stride layer:l
    :return:
    """
    m, nc, iw, ih = input_data.shape
    nf, nc, fw, fh = filters.shape

    ow, oh = (iw + 2*padding - fw) / stride + 1, (ih + 2*padding - fh) / stride + 1
    ow, oh = int(ow), int(oh)

    z = np.zeros((m, nf, ow, oh))
    # 加padding
    padding_input_data = np.pad(input_data, ((0, 0), (0, 0), (padding, padding), (padding, padding)),'constant')

    for index_m in range(m):   # 遍历batch中所有样本
        sample = padding_input_data[index_m]
        for index_w in range(ow):
            for index_h in range(oh):
                sample_slice = sample[:, index_w*stride:index_w*stride+fw, index_h*stride:index_h*stride+fh]    # 输入数据与filter对应的一个卷积块
                for index_f in range(nf):  # 遍历所有filter
                    z[index_m, index_f, index_w, index_h] = np.sum(sample_slice * filters[index_f]) + bias[index_f][0]

    a = np.maximum(z, 0)

    cache = (z, input_data, filters, bias, padding, stride)
    return a, cache


def conv_backword(da, cache):
    """
    :param da: 后一层传回的da 用于计算dz
    :param cache:
    :return:
    """
    # da: m batch大小, nf filter个数, ow, oh
    z, input_data, filters, bais, padding, stride = cache
    padding_a_1 = np.pad(input_data, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 'constant')
    # print(z.shape,padding_a_1.shape)

    # relu 求导
    z[z <= 0] = 0
    z[z > 0] = 1
    dz = da * z

    m,nf,w,h = dz.shape
    f = filters.shape[3]

    # m x nc x iw x ih
    da_1 = np.zeros(input_data.shape)
    dw = np.zeros(filters.shape)
    db = np.zeros(bais.shape)

    padding_da_1 = np.pad(da_1, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 'constant')
    for index_m in range(m):    # 遍历batch 样本
        for index_c in range(nf):
            for index_w in range(w):
                for index_h in range(h): # 遍历dz的所有通道,也是遍历所有filter
                    w_start = index_w * stride
                    w_end = index_w * stride + f
                    h_start = index_h * stride
                    h_end = index_h * stride + f

                    padding_da_1[index_m,:,w_start:w_end,h_start:h_end] += dz[index_m,index_c,index_w,index_h] * filters[index_c]
                    dw[index_c] += dz[index_m,index_c,index_w,index_h] * padding_a_1[index_m,:,w_start:w_end,h_start:h_end]
                    db[index_c] += dz[index_m,index_c,index_w,index_h]
        if padding:
            da_1[index_m, :, :, :] = padding_da_1[index_m,:,padding:-padding,padding:-padding]
        else:
            da_1[index_m, :, :, :] = padding_da_1[index_m]

    return da_1, dw, db


# def test_conv_2():
#     # np.random.seed(1)
#     input = np.random.randn(10,1,28,28).astype(np.float64)
#     w = np.random.randn(6,1,5,5).astype(np.float64)
#     b = np.random.randn(6,1).astype(np.float64)
#     y = np.ones((10,6,24,24)).astype(np.float64)
#     for i in range(50):
#         print('-'*50)
#         z, cache = conv_forward(input, w, b)
#         loss = np.mean(np.square(z - y))
#         print(loss)
#         dz = z-y
#         _, dw, db = conv_backword(dz, cache)
#         # print(dw)
#         # print(b)
#         # print(db)
#         w -= dw * 0.001
#         b -= db * 0.001


# test_conv_2()


# def test_conv():
#     np.random.seed(1)
#     x = np.array([x for x in range(1,17)] + [x for x in range(1,17)]).reshape(1,2,4,4)
#     w = np.ones((1, 2, 4, 4)) * 0.2
#     b = np.ones((1, 1))
#     print(x)
#     print("\n")
#
#     a, cache_conv = conv_forward(x, w, b)
#
#     print(a.shape)
#     print(a)
    # print("\n")
    # testa = np.ones((1,1,3,3))
    # print("\n")
    # da_1, dw, db = conv_backword(testa, cache_conv)
    # print(db)
    # print("\n")
    #
    # print(dw)
    # print("\n")
# test_conv()


def pool_forward(input_data, pool_size, stride, mode='max'):
    """
    数据经过 pooling layer, (m,nc)不变, (iw,ih)会变
    :param input_data: input  m x nc x iw x ih 前一层输出的
    :param pool_size: f
    :param stride:
    :param mode:
    :return: max or average
    """
    m, nc, iw, ih = input_data.shape
    ow, oh = int((iw-pool_size)/stride) + 1, int((ih-pool_size)/stride) + 1
    output = np.zeros((m, nc, ow, oh))
    for index_m in range(m):    # 遍历 batch 内数据
        sample = input_data[index_m]    # 当前数据
        for index_w in range(0, ow):
            for index_h in range(0, oh):
                for index_c in range(len(sample)):  # 遍历 channel
                    sample_slice = \
                        sample[index_c,index_w*stride:index_w*stride+pool_size, index_h*stride:index_h*stride+pool_size]
                    if mode == 'max':
                        sample_slice = np.max(sample_slice)
                    elif mode == 'avg':
                        sample_slice = np.mean(sample_slice)
                    output[index_m, index_c, index_w, index_h] = sample_slice
    cache = (input_data, pool_size, stride, mode)
    return output, cache


def pool_backward(da, cache):
    """
    :param da: (m x nc x w x h)
    :param cache: input_data , pool_size , stride
    :return:
    """
    # dz为后一层导数,经过pooling layer 传递到前一层
    # a_prev 为前一层的输出,也是当前pooling layer的输入
    a_prev, pool_size, stride, mode = cache
    da_m, da_nc, da_w, da_h = da.shape
    da_prev = np.zeros(a_prev.shape)

    # 遍历传入 da 的每一个值,并反向传递累加到dz_prev
    for index_m in range(da_m):
        for index_c in range(da_nc):
            for index_w in range(da_w):
                for index_h in range(da_h):
                    w_start = index_w * stride
                    w_end = index_w * stride + pool_size
                    h_start = index_h * stride
                    h_end = index_h * stride + pool_size
                    a_slice = a_prev[index_m,index_c,w_start:w_end,h_start:h_end]
                    if mode == 'max':
                        mask = a_slice == np.max(a_slice)
                        da_prev[index_m,index_c,w_start:w_end,h_start:h_end] += \
                            da[index_m,index_c,index_w,index_h] * mask
                    elif mode == 'avg':
                        da_prev[index_m, index_c, w_start:w_end, h_start:h_end] += \
                            da[index_m,index_c,index_w,index_h] * np.ones((pool_size,pool_size)) / (pool_size*pool_size)
    return da_prev


# def test_pool():
#     np.random.seed(1)
#     # A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
#     # A_prev = np.random.randn(2, 4, 4, 3)
#     A_prev = np.random.randn(1, 1, 6, 6)
#     A_prev = np.array(range(1,37)).reshape(1, 1, 6, 6)
#
#     pool = 2
#     stride = 2
#     res, cache = pool_forward(A_prev,pool,stride,'avg')
#     print(res)
#     print('-'*50)
#     # res = np.ones((1, 1, 3, 3))
#     res = pool_backward(res,cache)
#     print(res)
#     print(res.shape)
# test_pool()


def fc_forward(ai_1, w, b, mode):
    """
    :param ai_1: ai-1
    :param w: wi
    :param b: bi
    :return: ai-1, zi, ai, wi, bi
    """
    z = np.dot(w, ai_1) + b
    if mode == 'relu':
        ai = np.maximum(z, 0)
    else :
        if z.shape[1] == 1:
            exp = np.exp(z - np.max(z))
            ai = exp / np.sum(exp)
        else:
            exp = np.exp(z - np.max(z, axis=0, keepdims=True))
            ai = exp / np.sum(exp, axis=0, keepdims=True)
    cache = (ai_1, z, w)
    return ai, cache


def fc_backward(da, cache, dz=None):
    """
    :param da: dai 第 i 层 da 用于求dz
    :param dz: dzi 如传入dzi则函数内不再根据da,z计算dzi
    :return: dai-1 用于前一层求导计算
    """
    ai_1, z, w = cache
    if dz is None:
        z[z <= 0] = 0
        z[z > 0] = 1
        dz = da * z.T
    dw = np.dot(dz.T, ai_1.T)
    db = np.sum(dz.T, axis=1, keepdims=True)

    dai_1 = np.dot(dz, w)

    return dw,db,dai_1


def get_data():
    with open('../data/mnist_train.csv', 'r') as f:
    # with open('../data/mnist_test.csv', 'r') as f:
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


class LeNet:
    def __init__(self,learning_rate):
        self.batch_size = 64
        # self.w_a = np.random.randn(6, 1, 5, 5)/10
        # self.b_a = np.random.randn(6, 1)/10
        self.w_a = np.random.uniform(-1/6, 1/6, (6, 1, 5, 5))
        self.b_a = np.random.uniform(-1, 1, (6, 1))
        self.vdw_a = np.zeros(self.w_a.shape)
        self.vdb_a = np.zeros(self.b_a.shape)

        # self.w_c = np.random.randn(16, 6, 5, 5)/10
        # self.b_c = np.random.randn(16, 1)/10
        self.w_c = np.random.uniform(-1/6, 1/6, (16, 6, 5, 5))
        self.b_c = np.random.uniform(-1, 1,(16, 1))
        self.vdw_c = np.zeros(self.w_c.shape)
        self.vdb_c = np.zeros(self.b_c.shape)

        # self.w_e = np.random.randn(120, 16, 4, 4)/10
        # self.b_e = np.random.randn(120, 1)/10
        self.w_e = np.random.uniform(-1/10, 1/10, (120, 16, 4, 4))
        self.b_e = np.random.uniform(-1, 1,(120, 1))
        self.vdw_e = np.zeros(self.w_e.shape)
        self.vdb_e = np.zeros(self.b_e.shape)
        #
        # self.w_f = np.random.randn(84, 120)
        # self.b_f = np.zeros((84, 1))
        self.w_f = np.random.randn(84, 120)
        self.b_f = np.random.randn(84, 1)
        self.vdw_f = np.zeros(self.w_f.shape)
        self.vdb_f = np.zeros(self.b_f.shape)

        self.w_g = np.random.randn(10, 84)
        self.b_g = np.random.randn(10, 1)
        # self.w_g = np.random.uniform(-2.4/85, 2.4/85, (10, 84))
        # self.b_g = np.random.uniform(-2.4/85, 2.4/85, (10, 1))
        self.vdw_g = np.zeros(self.w_g.shape)
        self.vdb_g = np.zeros(self.b_g.shape)

        self.x, self.y = self.get_batch_data()
        self.learning_rate = learning_rate
        self.beta = 0.9

        print('learning_rate',self.learning_rate)

    @staticmethod
    def cross_entropy(p, y, epsilon=1e-12):
        p = np.clip(p, epsilon, 1. - epsilon)
        m = p.shape[1]
        ce = -np.sum(y * np.log(p)) / m
        return ce

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
        for index in range(len(x_batches)):
            batch_size = x_batches[index].shape[1]
            x_batches[index] = x_batches[index].T.reshape(batch_size,1,28,28)
        return x_batches,y_batches

    def test(self):
        x, y = get_test_data()
        m = x.shape[1]
        x = x.T
        x = x.reshape(m,1,28,28)
        true = 0
        for i in range(m):
            features = x[i].reshape(1,1,28,28)
            label = y[i]
            p = self.predict(features)
            print(p)
            p = p.tolist()
            pn = p.index(max(p))
            print(f'predict : {pn}, Ture: {label}')
            print('------')
            if pn == label:
                true += 1
        print(true/m *100)
            # break

    def predict(self,x):
        batch_size = x.shape[0]
        a_a, cache_a = conv_forward(x, self.w_a, self.b_a, padding=0, stride=1)
        a_b, cache_b = pool_forward(a_a, 2, 2, mode='avg')
        a_c, cache_c = conv_forward(a_b, self.w_c, self.b_c, padding=0, stride=1)
        a_d, cache_d = pool_forward(a_c, 2, 2, mode='avg')
        a_e, cache_e = conv_forward(a_d, self.w_e, self.b_e, padding=0, stride=1)
        a_e_reshape = a_e.reshape(batch_size, 120).T
        a_f, cache_f = fc_forward(a_e_reshape, self.w_f, self.b_f, 'relu')
        a_g, cache_g = fc_forward(a_f, self.w_g, self.b_g, 'softmax')
        return a_g

    def train(self,iteration):
        beta = self.beta
        for epoch in range(1,iteration+1):
            print('-'*50,epoch)
            self.learning_rate = self.learning_rate/epoch
            for x, y in zip(self.x, self.y):
                # print('=='*20)
                batch_size = x.shape[0]
                # --- forward -----
                # layer a convolution layer
                # input :  (batch-size, 1, 28, 28)
                # padding : 1
                # stride : 1
                # filters : (6, 1, 5, 5)
                # b : (6, 1)
                # output : (batch-size, 6, 24, 24)
                a_a, cache_a = conv_forward(x, self.w_a, self.b_a, padding=0, stride=1)
                # print('a',np.mean(a_a))

                # layer b pooling layer
                # input : (batch-size, 6, 24, 24)
                # stride : 2
                # filter : (2,2)
                # output : (batch-size, 6, 12, 12)
                # pool_forward(input_data, pool_size, stride, mode='max')
                a_b, cache_b = pool_forward(a_a, 2, 2, mode='avg')

                # layer c convolution layer
                # input : (batch-size, 6, 12, 12)
                # padding : 1
                # stride : 1
                # filters : (16, 6, 5, 5)
                # b : (16, 1)
                # output : (batch-size, 16, 8, 8)
                a_c, cache_c = conv_forward(a_b, self.w_c, self.b_c, padding=0, stride=1)
                # print('c',np.mean(a_c))

                # layer d pooling layer
                # input : (batch-size, 16, 8, 8)
                # stride : 2
                # filter : (2,2)
                # output : (batch-size, 16, 4, 4)
                a_d, cache_d = pool_forward(a_c, 2, 2, mode='avg')

                # layer e convolution layer
                # input : (batch-size, 16, 4, 4)
                # padding : 1
                # stride : 1
                # filters : (120, 16, 4, 4)
                # b : (120, 1)
                # output : (batch-size, 120, 1, 1)
                # reshape to (120*batch-size)
                a_e, cache_e = conv_forward(a_d, self.w_e, self.b_e, padding=0, stride=1)
                a_e_reshape = a_e.reshape(batch_size,120).T
                # print('e',np.mean(a_e))

                # layer f fully connected layer
                # input : (120, batch_size)
                # w : (84, 120)
                # b : (84, 1)
                # output : (84,batch-size)
                a_f, cache_f = fc_forward(a_e_reshape, self.w_f, self.b_f,'relu')
                # print('f',np.mean(a_f))

                # layer g output with softmax
                # input : (batch-size, 84)
                # w : (10, 84)
                # b : (10, 1)
                # output : (batch-size, 10)
                a_g, cache_g = fc_forward(a_f, self.w_g, self.b_g, 'softmax')

                loss = self.cross_entropy(a_g, y)
                # print("\n")
                print(f'loss : {loss}')
                # print("\n")

                # --- backward ---

                dz_g = (a_g - y).T / batch_size
                dw_g, db_g, da_f = fc_backward(None, cache_g, dz_g)

                # print('dwg',np.abs(dw_g).mean())
                # print('dbg',np.abs(db_g).mean())

                # self.w_g -= dw_g * self.learning_rate
                # self.b_g -= db_g * self.learning_rate
                vdw_g = beta * self.vdw_g + (1 - beta) * dw_g
                vdb_g = beta * self.vdb_g + (1 - beta) * db_g
                self.vdw_g = vdw_g
                self.vdb_g = vdb_g
                self.w_g -= vdw_g * self.learning_rate
                self.b_g -= vdb_g * self.learning_rate

                # layer f fully connected layer
                # input : (120.batch_size)
                # w : (84, 120)
                # b : (84, 1)
                # output : (84,batch-size)
                dw_f, db_f, da_e = fc_backward(da_f, cache_f, None)
                # print('dwf',np.abs(dw_f).mean())
                # print('dbf',np.abs(db_f).mean())

                # self.w_f -= dw_f * self.learning_rate
                # self.b_f -= db_f * self.learning_rate
                vdw_f = beta * self.vdw_f + (1 - beta) * dw_f
                vdb_f = beta * self.vdb_f + (1 - beta) * db_f
                self.vdw_f = vdw_f
                self.vdb_f = vdb_f
                self.w_f -= vdw_f * self.learning_rate
                self.b_f -= vdb_f * self.learning_rate

                # layer e convolution layer
                # input : (batch-size, 16, 4, 4)
                # padding : 1
                # stride : 1
                # filters : (120, 16, 4, 4)
                # b : (120, 1)
                # output : (batch-size, 120, 1, 1)
                # reshape to (120*batch-size)
                da_e_reshape = da_e.reshape(batch_size, 120, 1, 1)
                da_d, dw_e, db_e = conv_backword(da_e_reshape, cache_e)
                # print('dwe',np.abs(dw_e).mean())
                # print('dbe',np.abs(db_e).mean())

                # self.w_e -= dw_e * self.learning_rate
                # self.b_e -= db_e * self.learning_rate
                vdw_e = beta * self.vdw_e + (1 - beta) * dw_e
                vdb_e = beta * self.vdb_e + (1 - beta) * db_e
                self.vdw_e = vdw_e
                self.vdb_e = vdb_e
                self.w_e -= vdw_e * self.learning_rate
                self.b_e -= vdb_e * self.learning_rate

                # layer d pooling layer
                # input : (batch-size, 16, 8, 8)
                # stride : 2
                # filter : (2,2)
                # output : (batch-size, 16, 4, 4)
                da_c = pool_backward(da_d,cache_d)

                # layer c convolution layer
                # input : (batch-size, 6, 12, 12)
                # padding : 1
                # stride : 1
                # filters : (16, 6, 5, 5)
                # b : (16, 1)
                # output : (batch-size, 16, 8, 8)
                da_b,dw_c,db_c = conv_backword(da_c,cache_c)

                # print('dwc',np.abs(dw_c).mean())
                # print('dbc',np.abs(db_c).mean())

                # self.w_c -= dw_c * self.learning_rate
                # self.b_c -= db_c * self.learning_rate
                vdw_c = beta * self.vdw_c + (1 - beta) * dw_c
                vdb_c = beta * self.vdb_c + (1 - beta) * db_c
                self.vdw_c = vdw_c
                self.vdb_c = vdb_c
                self.w_c -= vdw_c * self.learning_rate
                self.b_c -= vdb_c * self.learning_rate

                # layer b pooling layer
                # input : (batch-size, 6, 24, 24)
                # stride : 2
                # filter : (2,2)
                # output : (batch-size, 6, 12, 12)
                # pool_forward(input_data, pool_size, stride, mode='max')
                da_a = pool_backward(da_b, cache_b)

                # layer a convolution layer
                # input :  (batch-size, 1, 28, 28)
                # padding : 1
                # stride : 1
                # filters : (6, 1, 5, 5)
                # b : (6, 1)
                # output : (batch-size, 6, 24, 24)
                da_x, dw_a, db_a = conv_backword(da_a, cache_a)
                # self.w_a -= dw_a * self.learning_rate
                # self.b_a -= db_a * self.learning_rate

                # print('dwa',np.abs(dw_a).mean())
                # print('dba',np.abs(db_a).mean())
                # print(dw_a)
                vdw_a = beta * self.vdw_a + (1 - beta) * dw_a
                vdb_a = beta * self.vdb_a + (1 - beta) * db_a
                self.vdw_a = vdw_a
                self.vdb_a = vdb_a
                self.w_a -= vdw_a * self.learning_rate
                self.b_a -= vdb_a * self.learning_rate

                # return


if __name__ == '__main__':
    net = LeNet(0.0001)
    net.train(5)
    net.test()


