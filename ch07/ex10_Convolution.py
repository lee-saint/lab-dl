import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from common.util import im2col, col2im
from dataset.mnist import load_mnist


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W  # weight: filter
        self.b = b  # bias
        self.stride = stride
        self.pad = pad

        # 중간 데이터: forward에서 생성되는 데이터 -> backward에서 사용
        self.x = None
        self.x_col = None
        self.W_col = None

        # gradients
        self.dW = None
        self.db = None

    def forward(self, x):
        """x: 4차원 이미지(mini-batch) 데이터"""
        self.x = x
        stride, pad = self.stride, self.pad
        n, c, h, w = x.shape
        fn, c, fh, fw = self.W.shape

        oh = (h - fh + 2 * pad) // stride + 1
        ow = (w - fw + 2 * pad) // stride + 1

        x_col = im2col(x, fh, fw, stride=stride, pad=pad)
        self.x_col = x_col
        w_col = self.W.reshape(fn, -1)
        self.W_col = w_col

        out = x_col.dot(w_col.T)
        out = out.reshape(n, oh, ow, -1).transpose(0, 3, 1, 2)
        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.x_col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.W_col.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


if __name__ == '__main__':
    np.random.seed(115)
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)
    print(x_train[:10].shape)

    W = np.random.randn(10, 1, 3, 3)
    # b = np.random.randn(10, 1, 1)
    b = np.zeros((10, 1, 1))
    conv = Convolution(W, b, stride=3, pad=1)

    x_conv = conv.forward(x_train[:10])
    print(x_conv.shape)

    plt.imshow(x_conv[0, 1], cmap='binary')
    plt.show()

    img = Image.open('sample2.jpg')
    img_arr = np.array(img)
    print(img_arr.shape)
    img_arr = img_arr.transpose(2, 0, 1).reshape(1, 3, 4000, 6000)
    print(img_arr.shape)

    W = np.random.randn(10, 3, 5, 5)
    b = np.random.randn(10, 1, 1)
    conv = Convolution(W, b, stride=5, pad=2)
    x_conv = conv.forward(img_arr)
    print(x_conv.shape)

    plt.imshow(x_conv[0, 4], cmap='inferno')
    plt.show()
