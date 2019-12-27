"""
mini-batch
"""
import pickle

import numpy as np

from ch03.ex01 import sigmoid
from dataset.mnist import load_mnist


def softmax(X):
    """
    1) X - 1차원: [x_1, x_2, ..., x_n]
    2) X - 2차원: [[x_11, x_12, ..., x_1n],
                   [x_21, x_22, ..., x_2n],
                   ...]
    """
    dimension = X.ndim
    if dimension == 1:
        m = np.max(X)  # 1차원 배열의 최댓값을 찾음
        X = X - m  # 0 이하의 숫자로 변환 <- exp 함수의 overflow를 방지하기 위해서
        y = np.exp(X) / np.sum(np.exp(X))
    elif dimension == 2:
        # m = np.max(X, axis=1).reshape((len(X), 1))
        # # len(X): 2차원 리스트 X의 row 개수
        # X = X - m
        # sum = np.sum(np.exp(X), axis=1).reshape((len(X), 1))
        # y = np.exp(X) / sum
        Xt = X.T  # X의 전치행렬(transpose)
        m = np.max(Xt, axis=0)
        Xt = Xt - m
        y = np.exp(Xt) / np.sum(np.exp(Xt), axis=0)
        y = y.T

    return y


def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    z1 = sigmoid(x.dot(W1) + b1)  # 첫번째 은닉층 전파(propagation)
    z2 = sigmoid(z1.dot(W2) + b2)  # 두번째 은닉층 전파(propagation)
    y = softmax(z2.dot(W3) + b3)  # 출력층 전파(propagation)
    return y


def mini_batch(network, X, batch_size):
    y_pred = np.array([])  # 예측값을 저장할 배열
    # batch_size 만큼씩 X의 데이터들을 나눠서 forward propagation(전파)
    for i in range(0, len(X), batch_size):
        X_batch = X[i:(i + batch_size)]
        y_hat = forward(network, X_batch)
        predictions = np.argmax(y_hat, axis=1)
        # 각 row에서 최댓값의 인덱스 -> (batch_size, ) 배열
        y_pred = np.append(y_pred, predictions)  # 예측값을 결과 배열에 추가
    return y_pred


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


if __name__ == '__main__':
    np.random.seed(2020)
    # 1차원 softmax 테스트
    a = np.random.randint(10, size=5)
    print(a)
    print(softmax(a))

    # 2차원 softmax 테스트
    A = np.random.randint(10, size=(2, 3))
    print(A)
    print(softmax(A))

    # [[2.83668596e-03 1.54877805e-01 4.21001524e-01]
    # [1.41230278e-04 1.41230278e-04 4.21001524e-01]]

    # (Train/Test) 데이터 세트 로드
    (X_train, y_train), (X_test, y_test) = load_mnist()
    print('X_test shape:', X_test.shape)  # (10000, 784)
    print('y_test shape:', y_test.shape)  # (10000, )
    print(X_test[0])
    print(y_test[0])

    # 신경망 생성 (W1, b1, ...)
    with open('sample_weight.pkl', mode='rb') as f:
        network = pickle.load(f)
    print('network:', network.keys())
    print('W1:', network['W1'].shape)
    print('W2:', network['W2'].shape)
    print('W3:', network['W3'].shape)

    batch_size = 100
    y_pred = mini_batch(network, X_test, batch_size)
    print('true[:10]', y_test[:10])
    print('pred[:10]', y_pred[:10])
    print('true[-10:]', y_test[-10:])
    print('pred[-10:]', y_pred[-10:])

    # 정확도(accuracy) 출력
    acc = accuracy(y_test, y_pred)
    print('정확도:', acc)
