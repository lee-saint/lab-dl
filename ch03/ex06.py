"""
MNIST 숫자 손글씨 데이터 세트
"""
from PIL import Image

from dataset.mnist import load_mnist
import numpy as np


def img_show(img_arr):
    """Numpy 배열(ndarray)로 작성된 이미지를 화면에 출력"""
    img = Image.fromarray(np.uint8(img_arr))  # NumPy 배열 형식을 이미지로 변환
    img.show()


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = load_mnist(normalize=False)
    # (학습 이미지 데이터, 학습 데이터 레이블),
    # (테스트 이미지 데이터 세트, 테스트 데이터 레이블)

    print('X_train shape:', X_train.shape)
    # (60000, 784): 28x28(=784) 크기의 이미지 60,000개
    print('y_train shape:', y_train.shape)
    # (60000, ): 60,000개 손글찌 이미지 숫자(레이블)

    # 학습 세트의 첫번째 이미지
    img = X_train[0]
    img = img.reshape((28, 28))  # 1차원 배열을 28x28 형태의 2차원 배열로 변환
    print(img)
    img_show(img)  # 2차원 NumPy 배열을 이미지로 출력
    print('label:', y_train[0])

    (X_train, y_train), (X_test, y_test) = load_mnist(normalize=True, flatten=False, one_hot_label=True)
    print('X_train shape:', X_train.shape)
    # flatten=False 인 경우 이미지 구성을 (컬러, 가로, 세로) 방식으로 표시
    # (60000, 1, 28, 28): 28x28 크기의 흑백 이미지 10,000개
    print('y_train shape:', y_train.shape)
    # one_hot_label=True 인 경우 one-hot encoding 형식으로 숫자 레이블 출력
    # (60000, 10)
    # 5 -> [0 0 0 0 0 1 0 0 0 0]
    # 0 -> [1 0 0 0 0 0 0 0 0 0]
    # 9 -> [0 0 0 0 0 0 0 0 0 1]
    print('y_train[0]:', y_train[0])

    # normalize=True인 경우, 각 픽셀에 숫자들이 0 ~ 1 사이의 숫자들로 정규화
    img = X_train[0]
    print(img)
