import os

from mxnet import nd
from mxnet import autograd
from mxnet import gluon

# 0. 生成数据

nun_inputs = 2
num_examples = 1000

true_w = [2, -4.2]
true_b = 3.14

X = nd.random.normal(scale=1.0, shape=(num_examples, nun_inputs))
y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] * true_w[1] + true_b

y += nd.random.normal(scale=0.01, shape=y.shape)

# 数据读取
batch_size = 10
dataset = gluon.data.ArrayDataset(X, y)
data_iter = gluon.data.DataLoader(dataset, batch_size, shuffle=True)

for data, label in data_iter:
    print(data, label)
    break

# 定义模型
net = gluon.nn.Sequential()

# 输出个数为1
net.add(gluon.nn.Dense(1))

# 初始化模型参数
from mxnet import init

net.initialize(init.Normal(sigma=0.01))

# 损失函数
loss = gluon.loss.L2Loss()

# 优化算法
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})

# 训练
# num_epochs = 5
# for epoch in range(num_epochs):
#     for data, label in data_iter:
#         with autograd.record():
#             l = loss(net(data), label)
#         l.backward()
#         trainer.step(batch_size)
#     l = loss(net(X), y).mean().asnumpy()
#     print("e : %d, loss : %f" % (epoch, l))

dense = net[0]

name_arr = [
    "课时1课件下载方法",
    "课时33Assignment 1 Q1- k-Nearest Neighbor classifier exercise",
    "课时34Assignment 1 Q2- Training a Support Vector Machine exercise",
    "课时35Assignment 1 Q3 - Implement a Softmax classifier exercise",
    "课时36Assignment 1 Q4 - Two-Layer Neural Network exercise",
    "课时37Assignment 1 Q5 - Image Features exercise",
    "课时38Assignment 2 Q1~Q3 - 搭建并训练一个强大的全连接神经网络",
    "课时39Assignment 2 Q4 - ConvNet on CIFAR-10 exercise",
    "课时40Assignment 2 Q5 - 手把手带你学会TensorFlow",
    "课时41Assignment 3 Q1~Q2 - Image Captioning (RNNs & LSTMs)",
    "课时42Assignment 3 Q3 - Saliency maps and Fooling Images exercise",
    "课时43Assignment 3 Q4 -  Image Generation：DeepDream exercise",
    "课时44Assignment 3 Q5 - 生成对抗网络（GANs）"
    ]

rootdir = '/Users/jack4c/Documents/xcache'
list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
name_list = []
for i in range(0, len(list)):
    # path = os.path.join(rootdir,list[i])
    path = list[i]
    if path.endswith("_0"):
        name_list.append(int(path.split("_")[0]))

name_list.sort()

for i, name in enumerate(name_list):
    print(name, name_arr[i])
    os.rename("/Users/jack4c/Documents/xcache/" + str(name) + "_0", "/Users/jack4c/Documents/xcache/" + name_arr[i]+".mp4")
