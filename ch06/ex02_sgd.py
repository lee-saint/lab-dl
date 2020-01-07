"""
신경망 학습 목적: 손실 함수의 값을 가능한 낮추는 파라미터를 찾는 것
파라미터(parameter):
    - 파라미터: 가중치(weight), 편향(bias)
    - 하이퍼 파라미터(hyper parameter): 학습률, epoch, batch 크기, 신경망 층에서의 뉴런 개수, 신경망 은닉층 개수
ch06의 목표:
    - 파라미터를 갱신하는 방법: SGD, Momentum, AdaGrad, Adam
    - 하이퍼 파라미터를 최적화시키는 방법
"""
import matplotlib.pyplot as plt
import numpy as np

from ch06.ex01_matplot3d import fn_derivative, fn


class Sgd:
    """SGD: Stochastic Gradient Descent(확률적 경사 하강법)
    W = W - lr * dL/dW
    W: 파라미터(가중치, 편향)
    lr: 학습률(learning rate
    """

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, params, gradients):
        """파라미터 params와 변화율 gradients가 주어지면 파라미터를 갱신하는 메소드
        params, gradients: 딕셔너리 {key: value, ...}
        """
        for key in params:
            # W = W - lr * dL/dW
            params[key] -= self.learning_rate * gradients[key]


if __name__ == '__main__':
    # Sgd 클래스의 객체(인스턴스)
    sgd = Sgd(learning_rate=0.95)  # 0.01, 0.1, 0.95

    # ex01 모듈에서 작성한 fn(x, y) 함수의 최솟값을 임의의 점에서 시작해서 찾아감
    init_position = (-7, 2)
    params = dict()
    params['x'], params['y'] = init_position

    # 각 파라미터에 대한 변화율(gradient)
    gradients = dict()
    gradients['x'], gradients['y'] = 0, 0

    # 각 파라미터(x, y)를 갱신할 때마다 갱신된 값을 저장할 리스트
    x_history = []
    y_history = []
    for i in range(30):
        x_history.append(params['x'])
        y_history.append(params['y'])
        gradients['x'], gradients['y'] = fn_derivative(params['x'], params['y'])
        sgd.update(params, gradients)

    for x, y in zip(x_history, y_history):
        print(f'({x}, {y})')

    # f(x, y) 함수를 등고선으로 표현
    x = np.linspace(-10, 10, 200)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = fn(X, Y)

    mask = Z > 7
    Z[mask] = 0

    plt.contour(X, Y, Z, 10)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')

    # 등고선 그래프에 파라미터(x, y)가 갱신되는 과정을 추가
    plt.plot(x_history, y_history, 'o-', color='red')
    plt.show()
