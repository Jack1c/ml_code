from mxnet import ndarray as nd
import matplotlib.pyplot as plt
import numpy as np


# 预测函数
def h(X, params):
    return params[0] + params[1] * X


# 损失函数
def J(params, X, Y):
    m = len(X)
    return np.sum(((h(X, params) - Y) ** 2)) / (2 * m)


def J_2(params, X, Y):
    m = len(X)
    return nd.sum(((h(X, params) - Y) ** 2)) / (2 * m)


# 梯度下降
def SGD(lr, params, X, Y):
    m = len(X)
    tmp_0 = params[0] - lr * np.sum(h(X, params) - Y) / m
    tmp_1 = params[1] - lr * np.sum((h(X, params) - Y) * X) / m
    return np.array([tmp_0, tmp_1])


def SGD2(lr, params):
    for param in params:
        param[:] = param - lr * param


# 训练
def train(X, Y, params, lr):
    for param in params:
        param.attach_grad()
    for i in range(100000):
        with autograd.record():
            l = J_2(params, X, Y)
        l.backward()
        params = SGD2(lr, params)

        if (i % 10000 == 0):
            print(J_2(params, X, Y))
    return params


from mxnet import nd
from mxnet import autograd


def main():
    # 真是参数
    true_params = nd.array([2])
    true_b = nd.array([-3.14])

    true_params = []
    X = nd.random.normal(scale=1, shape=(1000,)).asnumpy()
    Y = h(X, true_params) + nd.normal(scale=0.01, shape=(1000,)).asnumpy()

    X = nd.array(X.tolist())
    Y = nd.array(Y.tolist())

    params = nd.zeros(shape=(2,))



    lr = 0.001
    params = train(X, Y, params, lr)
    print("out put params :", params)
    print("true params :", true_params)


if __name__ == '__main__':
    main()
