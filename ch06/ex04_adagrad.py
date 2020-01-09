"""
파라미터 최적화 방법 3) AdaGrad(Adaptive Gradient)
    SGD (W = W - lr * grad) 에서는 학습률이 고정되어 있음
    AdaGrad에서 학습률을 변화시키면서 파라미터를 최적화함
    처음에는 큰 학습률로 시작, 점점 학습률을 줄여나가면서 파라미터 갱신
    h = h + grad * grad
    lr = lr / sqrt(h)
    W = W - (lr/sqrt(h)) * grad
"""
import matplotlib.pyplot as plt
import numpy as np

from ch06.ex01_matplot3d import fn, fn_derivative


class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr     # 학습률
        self.h = dict()  # 학습률을 변화시킬 때 사용할 파라미터 - lr/sqrt(h)

    def update(self, params, gradients):
        if not self.h:  # empty dict이면
            for key in params:
                # 파라미터와 shape가 같은 0으로 채워진 배열을 dict에 저장
                self.h[key] = np.zeros_like(params[key])
        for key in params:
            # h = h + grad * grad
            self.h[key] += gradients[key] * gradients[key]
            # W = W - (lr/sqrt(h)) * grad
            epsilon = 1e-8  # 0으로 나누는 것을 방지
            params[key] -= self.lr * gradients[key] / (np.sqrt(self.h[key]) + epsilon)


if __name__ == '__main__':
    adagrad = AdaGrad(lr=1.5)  # lr=0.01, 0.1, 0.95, 1.5

    params = {'x': -7., 'y': 2.}
    gradients = {'x': 0, 'y': 0}
    x_history = []
    y_history = []

    for _ in range(30):
        x_history.append(params['x'])
        y_history.append(params['y'])
        gradients['x'], gradients['y'] = fn_derivative(params['x'], params['y'])
        adagrad.update(params, gradients)

    for x, y in zip(x_history, y_history):
        print(f'({x}, {y})')

    x = np.linspace(-10, 10, 2000)
    y = np.linspace(-5, 5, 1000)
    X, Y = np.meshgrid(x, y)
    Z = fn(X, Y)

    mask = Z > 7
    Z[mask] = 0

    plt.contour(X, Y, Z, 10)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')

    plt.plot(x_history, y_history, 'o-', color='red')
    plt.show()
