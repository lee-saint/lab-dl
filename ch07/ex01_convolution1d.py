"""
1차원 Convolution(합성곱), Cross-Correlation(교차상관) 연산
"""
import numpy as np


def convolution_1d(x, w):
    """x, w: 1d ndarray, len(x) >= len(w)
    x와 w의 합성곱 결과를 리턴"""
    w_r = np.flip(w)
    return cross_correlation_1d(x, w_r)


def cross_correlation_1d(x, w):
    """x, w: 1d ndarray, len(x) >= len(w)
    x와 w의 교차 상관(cross-correlation) 연산 결과를 리턴"""
    # -> convolution_1d() 함수가 cross_correlation_1d()를 사용하도록 수정
    return np.array([np.sum(x[i:i + len(w)] * w) for i in range(len(x) - len(w) + 1)])


if __name__ == '__main__':
    np.random.seed(113)
    x = np.arange(1, 6)
    print('x =', x)
    w = np.array([2, 1])
    print('w =', w)

    # Convolution(합성곱) 연산
    # 1) w를 반전
    # w_r = np.array([1, 2])
    w_r = np.flip(w)
    print('w_r =', w_r)

    # 2) FMA(Fused Multiply-Add)
    conv = []
    for i in range(4):
        x_sub = x[i:i+2]  # (0,1), (1,2), (2,3), (3,4)
        fma = np.dot(x_sub, w_r)
        conv.append(fma)
    conv = np.array(conv)
    print(conv)
    # 1차원 convolution 연산 결과의 크기(원소의 개수) = len(x) - len(w) + 1

    # convolution_1d 함수 테스트
    conv = convolution_1d(x, w)
    print(conv)

    x = np.arange(1, 6)
    w = np.array([2, 0, 1])
    conv = convolution_1d(x, w)
    print(conv)

    # 교차 상관(Cross-Correlation) 연산
    # 합성곱 연산과 다른 점은 w를 반전시키지 않는다는 것
    # CNN(Convolutional Neural Network, 합성곱 신경망)에서는 가중치 행렬을 난수로 생성 후 GD 등을 사용해 갱신하기 때문에
    # 대부분의 경우 합성곱 연산 대신 교차 상관을 사용함
    cross_corr = cross_correlation_1d(x, w)
    print(cross_corr)
