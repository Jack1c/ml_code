from mxnet import nd
from mxnet import autograd

# 0. 生成数据

num_inputs = 2
num_examples = 1000

true_w = [2, -3.4]
true_b = 4.2

X = nd.random.normal(scale=1.0, shape=(num_examples, num_inputs))
y = X[:, 0] * true_w[0] + X[:, 1] * true_w[1] + true_b
y += nd.random.normal(scale=0.01, shape=y.shape)

batch_size = 10

## 1.读取数据

import random


def data_iter(X, y, batch_size):
    length = len(X)
    indexs = list(range(length))
    for i in range(0, length, batch_size):
        j = nd.array(indexs[i: min(i + batch_size, length)])
        yield X.take(j), y.take(j)


## 2. 定义模型
def net(X, w, b):
    return nd.dot(X, w) + b


## 3. 初始化模型参数
w = nd.random.normal(scale=0.01, shape=(num_inputs, 1))
b = nd.zeros(shape=(1,))

## 给参数开梯度
w.attach_grad()
b.attach_grad()


## 4.平方差的损失函数
def loss(y_hate, y):
    return (y_hate - y.reshape(y_hate.shape)) ** 2 / 2


## 5.优化
def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - param.grad * lr / batch_size


## 训练

lr = 0.01
num_epochs = 5

for epoch in range(num_epochs):
    for data, label in data_iter(X, y, batch_size):
        with autograd.record():
            yhate = net(data, w, b)
            _loss = loss(yhate, label)
        _loss.backward()
        sgd([w, b], lr, batch_size)
    print("total loss :", nd.sum(loss(net(X, w, b), y)).asscalar() / num_examples)

print(true_w, true_b)
print(w, b)
