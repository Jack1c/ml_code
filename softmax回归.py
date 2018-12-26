import gluonbook as gb
from mxnet.gluon import data as gdata

import sys
import time

minst_train = gdata.vision.FashionMNIST(train=True)
minst_test = gdata.vision.FashionMNIST(train=False)

len(minst_test)
len(minst_train)

feature, label = minst_train[0]

print(feature.shape, feature.dtype)
print(label.shape, label.dtype)


def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_fashion_mnist(images, labels):
    gb.use_svg_display()
    _, figs = gb.plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.reshape((28, 28)).asnumpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    gb.plt.show()


X, y = minst_train[0:9]

show_fashion_mnist(images=X, labels=y)

batch_size = 256

