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


# 梯度下降
def SGD(lr, params, X, Y):
    m = len(X)
    tmp_0 = params[0] - lr * np.sum(h(X, params) - Y) / m
    tmp_1 = params[1] - lr * np.sum((h(X, params) - Y) * X) / m
    return np.array([tmp_0, tmp_1])


# 训练
def train(X, Y, params, lr):
    for i in range(100000):
        params = SGD(lr, params, X, Y)
        if(i % 10000 == 0):
            print(J(params, X, Y))
    return params


def main():
    # 真是参数
    true_params = [2, -3.4]

    X = np.random.rand(10000)
    Y = h(X, true_params)

    params = np.zeros(2)
    lr = 0.001
    params = train(X, Y, params, lr)
    print("out put params :", params)
    print("true params :", true_params)


if __name__ == '__main__':
    main()
