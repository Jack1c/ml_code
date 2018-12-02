from mxnet import gluon, nd

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2

features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b

labels += nd.random.normal(scale=0.01, shape=labels.shape)

# 数据读取
from mxnet.gluon import data as gdata

batch_size = 10

data_set = gdata.ArrayDataset(features, labels)
data_iter = gdata.DataLoader(data_set, batch_size, shuffle=True)

# for x, y in data_iter:
#     print(x, y)
#     break

from mxnet.gluon import nn

# nn neural networks
# nn 中定义了大量的神经网络层 , Sequential 串联神经网络层的容器
net = nn.Sequential()

# 线性回归中的输出层又叫做全连接层 为一个Dense实例
net.add(nn.Dense(1))

# 初始模型参数
# 在使用net前 需要先初始化模型参数, mxnet中导入 initializer 模块

from mxnet import init

net.initialize(init.Normal(sigma=0.01))

# 损失函数
from mxnet.gluon import loss as gloss

loss = gloss.L2Loss()

# 优化算法
from mxnet import gluon

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})

from mxnet import autograd

# 训练模型
num_epochs = 3

for epoch in range(num_epochs):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    l = loss(net(X), y)
    print('epoch %d, loss: %f' % (epoch, l.mean().asnumpy()))

dense = net[0]
print(true_w, dense.weight.data())

print(true_b, dense.bias.data())
