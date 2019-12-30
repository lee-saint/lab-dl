"""
weight 행렬에 경사 하강법(gradient descent) 적용
"""
import numpy as np

from ch03.ex11 import softmax
from ch04.ex03 import cross_entropy
from ch04.ex05 import numerical_gradient


class SimpleNetwork:
    def __init__(self):
        np.random.seed(1230)
        self.W = np.random.randn(2, 3)
        # 가중치 행렬 (2x3 행렬)의 초깃값들을 임의로 설정

    def predict(self, x):
        z = x.dot(self.W)
        y = softmax(z)
        return y

    def loss(self, x, y_true):
        """손실 함수(loss function)"""
        y_pred = self.predict(x)  # 입력이 y일 때 출력 y의 예측값 계산
        ce = cross_entropy(y_pred, y_true)
        return ce

    def gradient(self, x, t):
        """x: 입력, t: 출력 실제 값(정답 레이블)"""
        fn = lambda W: self.loss(x, t)
        return numerical_gradient(fn, self.W)


if __name__ == '__main__':
    # SimpleNetwork 클래스 객체를 생성
    network = SimpleNetwork()  # 생성자 호출 -> __init__() 메소드 호출
    print('W =', network.W)
    print(id(network.W))

    # x = [0.6, 0.9]일 때 y_true = [0 0 1]이라고 가정
    x = np.array([0.6, 0.9])
    y_true = np.array([0.0, 0.0, 1.0])
    print('x =', x)
    print('y_true =', y_true)

    y_pred = network.predict(x)
    print('y_pred =', y_pred)

    ce = network.loss(x, y_true)
    print('cross entropy =', ce)

    g1 = network.gradient(x, y_true)
    print('g1 =', g1)

    # grads = np.zeros_like(network.W)
    # # print(grads)
    #
    # for idx, z in enumerate(network.W):
    #     # print(idx, z)
    #     grad = np.zeros_like(z)
    #     # print(grad)
    #     h = 1e-4
    #     for i in range(z.size):
    #         ith_value = z[i]
    #         # print(ith_value)
    #         z[i] = ith_value + h
    #         fh1 = network.loss(x, y_true)
    #         z[i] = ith_value - h
    #         fh2 = network.loss(x, y_true)
    #         grad[i] = (fh1 - fh2) / (2 * h)
    #         z[i] = ith_value
    #     grads[idx] = grad
    #
    # print('gradient =', grads)

    lr = 0.1  # learning rate
    network.W -= lr * g1
    print('W =', network.W)
    print('y_pred =', network.predict(x))
    print('ce =', network.loss(x, y_true))

    for i in range(200):
        print(f'{i}th trial..........')
        g1 = network.gradient(x, y_true)
        print('g1 =', g1)
        network.W -= lr * g1
        print('W =', network.W)
        print('y_pred =', network.predict(x))
        print('ce =', network.loss(x, y_true))

