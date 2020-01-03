"""
sigmoid 함수: y = 1 / (1 + exp(-x))
dy/dx = y(1-y) 증명
sigmoid 뉴런을 작성(forward, backward)
"""
import numpy as np

from ch03.ex01 import sigmoid


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
    # Sigmoid 뉴런을 생성
    sigmoid_gate = Sigmoid()
    # x = 0일 때 sigmoid 함수의 리턴값(forward)
    y = sigmoid_gate.forward(0.)
    print('y =', y)  # x = 0 -> sigmoid(0) = 0.5

    # x = 0일 때 sigmoid의 gradient(접선의 기울기)
    dx = sigmoid_gate.backward(1.)
    print('dx =', dx)

    # 아주 작은 h에 대해서 (f(x + h) - f(x - h)) / (2 * h)를 계산
    h = 1e-4
    dx2 = (sigmoid(0. + h) - sigmoid(0. - h)) / (2 * h)
    print('dx2 =', dx2)
