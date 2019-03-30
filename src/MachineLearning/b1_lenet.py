import numpy as np


def conv_forward(input_data, filters, bais=None, padding=0, stride=1):
    """
    :param input_data: input  m x nc x iw x ih
    :param filters: filter nf x nc x fw x fh  n为filter数量
    :param bais: bias 每一filter对应一个实数b nf x 1
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
                temp = sample[:, index_w*stride:index_w*stride+fw, index_h*stride:index_h*stride+fh]    # 输入数据与filter对应的一个卷积块
                for index_f in range(nf):  # 遍历所有filter
                    z[index_m, index_f, index_w, index_h] = np.sum(temp * filters[index_f] + bais[index_f])
    a = np.maximum(z, 0)
    cache = (z, input_data, filters, bais, padding, stride)
    # return a, cache
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
    z[z > 0] = 1
    dz = da*z

    m,nf,w,h = dz.shape
    f = filters.shape[3]

    # m x nc x iw x ih
    da_1 = np.zeros(input_data.shape)
    dw = np.zeros(filters.shape)
    db = np.zeros(bais.shape)

    padding_da_1 = np.pad(da_1, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 'constant')

    for index_m in range(m):    # 遍历batch 样本
        for index_w in range(w):
            for index_h in range(h):
                for index_c in range(nf): # 遍历dz的所有通道,也是遍历所有filter
                    w_start = index_w * stride
                    w_end = index_w * stride + f
                    h_start = index_h * stride
                    h_end = index_h * stride + f

                    padding_da_1[index_m,:,w_start:w_end,h_start:h_end] += dz[index_m,index_c,index_w,index_h] * filters[index_c]
                    dw[index_c] += dz[index_m,index_c,index_w,index_h] * padding_a_1[index_m,:,w_start:w_end,h_start:h_end]

                    db[index_c] += dz[index_m,index_c,index_w,index_h]

        da_1[index_m, :, :, :] = padding_da_1[index_m,:,padding:-padding,padding:-padding]
    return da_1, dw, db


def test_conv():
    np.random.seed(1)
    # A_prev -- output activations of the previous layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    # A_prev = np.random.randn(10, 4, 4, 3)
    # A_prev = np.random.randn(10, 3, 4, 4)
    A_prev = np.ones((10, 3, 4, 4))
    #  W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
    # W = np.random.randn(2, 2, 3, 8)
    # W = np.random.randn(8, 3, 2, 2)
    W = np.ones((8, 3, 2, 2))
    b = np.random.randn(8,1)

    z, cache_conv = conv_forward(A_prev, W, b, 2,1)
    # print(Z[0][0])
    print("Z's mean =", np.mean(z))
    # print("cache_conv[0][1][2][3] =", cache_conv[0][1][2][3])
    np.random.seed(1)
    da_1, dw, db = conv_backword(z, cache_conv)
    print("dA_mean =", np.mean(da_1))
    print("dW_mean =", np.mean(dw))
    print("db_mean =", np.mean(db))
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


def test_pool():
    np.random.seed(1)
    # A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    # A_prev = np.random.randn(2, 4, 4, 3)
    A_prev = np.random.randn(1, 1, 4, 4)
    print(A_prev)
    pool = 2
    stride = 2
    res, cache = pool_forward(A_prev,pool,stride,'max')
    print(res)
    print('-'*50)
    res = pool_backward(res,cache)
    print(res)
test_pool()













