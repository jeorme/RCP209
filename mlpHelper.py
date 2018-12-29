import numpy as np


def forwardMLP(batch, Wh, bh, Wy, by):
    h = np.matmul(batch, Wh) + bh
    h = sigmoid(h)
    return h, forward(h, Wy, by)


def forward(batch, W, b):
    """compute y predicted from W and b and a batch in the case of softmax"""
    s = np.matmul(batch, W) + b
    return softmax(s)


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def softmax(x):
    ex = np.exp(x)
    return ex / np.sum(ex, axis=1)[:, None]


def computeGradient(y_train, y_predict, x_train, batch_size):
    return np.matmul(np.transpose(x_train), (y_predict - y_train)) / batch_size, np.sum(
        y_predict - y_train) / batch_size


def computeGradientHidden(y_train, y_predict, wy,x_train, hidden,batch_size):
    deltah =np.matmul((y_train - y_predict),np.transpose(wy)) * hidden*(1-hidden)
    return np.matmul(np.transpose(x_train), deltah) / batch_size, np.sum(
        deltah) / batch_size


def accuracy(W, b, images, labels):
    pred = forward(images, W, b)
    return np.where(pred.argmax(axis=1) != labels.argmax(axis=1), 0., 1.).mean() * 100.0

def accuracyMLP(Wh, bh,Wy, by,  images, labels):
    pred,hidden =  forwardMLP(images, Wh, bh, Wy, by)
    return np.where(pred.argmax(axis=1) != labels.argmax(axis=1), 0., 1.).mean() * 100.0
