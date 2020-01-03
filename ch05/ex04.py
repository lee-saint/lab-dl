"""
sigmoid 함수: y = 1 / (1 + exp(-x))
dy/dx = y(1-y) 증명
sigmoid 뉴런을 작성(forward, backward)
"""
import numpy as np


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        result = 1 / (1 + np.exp(-x))
        self.out = result
        return result

    def backward(self, delta):
        return delta * self.out * (1 - self.out)


if __name__ == '__main__':
    pass
