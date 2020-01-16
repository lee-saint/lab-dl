import matplotlib.pyplot as plt
import numpy as np

from common.util import im2col, col2im
from dataset.mnist import load_mnist


class Pooling:
    def __init__(self, fh, fw, stride=1, pad=0):
        self.fh = fh                  # pooling 윈도우의 높이(height)
        self.fw = fw                  # pooling 윈도우의 너비(width)
        self.stride = stride          # pooling 윈도우를 이동시키는 크기(보폭)
        self.pad = pad                # 패딩 크기

        # backward에서 사용하게 될 값
        self.x = None                 # pooling 레이어로 forward 되는 데이터
        self.arg_max = None           # 찾은 최댓값의 인덱스

    def forward(self, x):
        """x: (samples, channel, height, width) 모양의 4차원 배열"""
        self.x = x
        n, c, h, w = x.shape
        fh, fw, stride, pad = self.fh, self.fw, self.stride, self.pad
        oh, ow = (h + 2 * pad - fh) // stride + 1, (w + 2 * pad - fw) // stride + 1

        col = im2col(x, fh, fw, stride, pad)
        col = col.reshape(-1, fh*fw)
        out = np.max(col, axis=1)
        self.arg_max = np.argmax(col, axis=1)
        out = out.reshape(n, oh, ow, -1).transpose(0, 3, 1, 2)
        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.fh * self.fw
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size, ))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.fh, self.fw, self.stride, self.pad)

        return dx


if __name__ == '__main__':
    np.random.seed(116)

    # Pooling 클래스의 forward 메소드를 테스트
    # x를 (1, 3, 4, 4) 모양으로 무작위로 생성, 테스트
    x = np.random.randint(10, size=(1, 3, 4, 4))
    print(x, 'shape:', x.shape)
    pool = Pooling(2, 2, stride=2)
    pooled = pool.forward(x)
    print(pooled, 'shape:', pooled.shape)

    # MNIST 데이터를 로드
    # 학습 데이터 중 5개를 batch로 forward
    # forwarding된 결과(pooling 결과)를 pyplot으로 그림

    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=False, flatten=False)
    x_batch = x_train[:5]
    print(x_batch.shape)

    pool = Pooling(4, 4, stride=4)
    pooled_batch = pool.forward(x_batch)
    print(pooled_batch.shape)
    # fig, axes = plt.subplots(3, 2, sharex=True, sharey=True)
    # flat_axes = axes.flatten()

    fig, axes = plt.subplots(2, 5)
    for i in range(len(axes[0])):
        axes[0, i].imshow(x_batch[i].squeeze(), cmap='gray')
        axes[1, i].imshow(pooled_batch[i].squeeze(), cmap='gray')
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
    plt.show()
