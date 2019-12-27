"""
Perceptron:
    - 입력: (x1, x2)
    - 출력:
        a = x1 * w1 + x2 * w2 + b 계산
        y = 1, a > 임계값
          = 0, a <= 임계값
신경망의 뉴런(neuron)에서는 입력 신호의 가중치 합을 출력값으로 변환해주는 함수가 존재
-> 활성화 함수(activation function)
"""
import matplotlib.pyplot as plt
import numpy as np


def step_function(x):
    y = x > 0  # [False, False, ..., True]
    return y.astype(np.int)


def sigmoid(x):
    """sigmoid = 1 / (1 + exp(-x))"""
    return 1 / (1 + np.exp(-x))


def relu(x):
    """ReLU(Rectified Linear Unit)
        y = x, if x > 0
          = 0, otherwise
    """
    return np.maximum(x, 0)


if __name__ == '__main__':
    x = np.arange(-3, 4)
    print('x =', x)
    # for x_i in x:
    #     print(step_function(x_i), end=' ')
    # print()
    print('y =', step_function(x))  # [0 0 0 0 1 1 1]

    print('sigmoid =', sigmoid(x))

    # step 함수, sigmoid 함수를 하나의 그래프에 출력
    x = np.arange(-10, 10, 0.01)
    steps = step_function(x)
    sigmoids = sigmoid(x)
    plt.plot(x, steps, label='Step Function')
    plt.plot(x, sigmoids, label='Sigmoid Function')
    plt.legend()
    plt.show()

    x = np.arange(-3, 4)
    print('x =', x)
    relus = relu(x)
    print('relu =', relus)
    plt.plot(x, relus)
    plt.title('ReLU')
    plt.show()
