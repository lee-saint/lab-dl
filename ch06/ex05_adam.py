"""
파라미터 최적화 알고리즘 4) Adam(Adaptive Moment estimate)
    AdaGrad + Momentum 알고리즘
    학습률 변화 + 속도(모멘텀) 개념 도입
    t: timestamp / 반복할 때마다 증가하는 숫자. update 메소드가 호출될 때마다 +1
    beta1, beta2: 모멘텀을 변화시킬 때 사용하는 상수
    m: 1st momentum
    v: 2nd momentum
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad * grad
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    W = W - lr * m / (sqrt(v) + eps)
"""
import matplotlib.pyplot as plt
import numpy as np

from ch06.ex01_matplot3d import fn, fn_derivative


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.b1 = beta1
        self.b2 = beta2
        self.m = dict()
        self.v = dict()
        self.m_hat = dict()
        self.v_hat = dict()
        self.t = 0

    def update(self, params, grads, eps=1e-8):
        # m, v 초기화
        if not self.m:
            for key in params:
                self.m[key] = np.zeros_like(params[key])
        if not self.v:
            for key in params:
                self.v[key] = np.zeros_like(params[key])

        # timestamp 증가
        self.t += 1

        for key in params:
            self.m[key] = self.b1 * self.m[key] + (1 - self.b1) * grads[key]
            self.v[key] = self.b2 * self.v[key] + (1 - self.b2) * grads[key] * grads[key]
            self.m_hat[key] = self.m[key] / (1 - self.b1**self.t)
            self.v_hat[key] = self.v[key] / (1 - self.b2**self.t)
            params[key] -= self.lr * self.m_hat[key] / (np.sqrt(self.v_hat[key]) + eps)


if __name__ == '__main__':
    adam = Adam(lr=0.3, beta1=0.9, beta2=0.99)

    params = {'x': -7., 'y': 2.}
    gradients = {'x': 0, 'y': 0}
    x_history = []
    y_history = []

    for _ in range(100):
        x_history.append(params['x'])
        y_history.append(params['y'])
        gradients['x'], gradients['y'] = fn_derivative(params['x'], params['y'])
        adam.update(params, gradients)

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
