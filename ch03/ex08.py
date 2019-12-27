"""
MNIST 숫자 손글씨 데이터 신경망 구현
"""
import pickle

import numpy as np

from ch03.ex01 import sigmoid
from ch03.ex05 import forward, softmax
from ch03.ex06 import img_show
from dataset.mnist import load_mnist


def init_network():
    """가중치 행렬(W1, W2, W3, b1, b2, b3)을 생성"""
    # 저자가 만든 가중치 행렬 읽어옴
    with open('sample_weight.pkl', mode='rb') as f:
        network = pickle.load(f)
    return network


def forward(network, x):
    """
    순방향 전파(pro-pagation)
    :param network:
    :param x: 하나의 이미지 정보를 가지고 있는 배열 (784, )
    """
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    # 첫번째 은닉층
    a1 = x.dot(W1) + b1
    z1 = sigmoid(a1)

    # 두번째 은닉층
    a2 = z1.dot(W2) + b2
    z2 = sigmoid(a2)

    # 출력층
    a3 = z2.dot(W3) + b3
    y = softmax(a3)
    return y


def predict(network, X_test):
    """신경망에서 사용되는 가중치 행렬과 테스트 데이터를 파라미터로 받아 테스트 데이터의 예측값(배열)을 리턴
    X_test: 10,000개의 테스트 이미지 정보를 가지고 있는 배열"""
    y_pred = []
    for sample in X_test:  # 테스트 세트의 각 이미지에 대해 반복
        sample_hat = forward(network, sample)
        # 가장 큰 확률의 인덱스(-> 예측값)를 찾음
        sample_pred = np.argmax(sample_hat)
        y_pred.append(sample_pred)  # 예측값을 결과 리스트에 추가
    return y_pred


def accuracy(y_test, y_pred):
    """테스트 데이터 레이블과 예측값을 파라미터로 받아
    정확도(accuracy) = (정답 개수) / (테스트 데이터 개수) 를 리턴"""
    right = np.sum(np.equal(y_test, y_pred))
    every = len(y_test)
    return right / every


if __name__ == '__main__':
    # 데이터 준비(학습 세트, 테스트 세트)
    (X_train, y_train), (X_test, y_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)

    # 신경망 가중치(와 편향, bias) 행렬들 생성
    network = init_network()
    for key in list(network.keys()):
        print(f'{key} shape: {network[key].shape}')

    y_pred = predict(network, X_test)
    print(y_pred[:10])
    acc = accuracy(y_test, y_pred)
    print(acc)
    # wrong = np.argwhere(y_test != y_pred)[6:11]
    # for idx in wrong:
    #     print(y_test[idx])
    #     img = (X_test[idx] * 255).reshape((28, 28))
    #     img_show(img)

