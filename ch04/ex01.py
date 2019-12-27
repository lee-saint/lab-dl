"""
Machine Learning(기계 학습) -> Deep Learning(심층 학습)
training data set(학습 세트) / test data set(검증 세트)
신경망 층을 지나갈 때 사용되는 가중치(weight) 행렬, 편항(bias) 행렬을 찾는 게 목적
오차를 최소화하는 가중치 행렬을 찾아야 한다
손실(loss) 함수 / 비용(cost) 함수의 값을 최소화하는 가중치 행렬 찾기
손실 함수:
    - 평균 제곱 오차(MSE: Mean Squared Error)
    - 교차 엔트로피(Cross-Entropy)
"""
import numpy as np
from dataset.mnist import load_mnist

if __name__ == '__main__':
    (X_train, y_train), (X_test, y_true) = load_mnist()
    # 10개 테스트 이미지의 실제 값
    print('y_true =', y_true[:10])

    # 10개 테스트 이미지의 예측 값
    y_pred = np.array([7, 2, 1, 6, 4, 1, 4, 9, 6, 9])
    print('y_pred =', y_pred)

    # 오차
    error = y_pred - y_true[:10]
    print('error =', error)

    # 오차 제곱(squared error)
    sq_err = error ** 2
    print('squared error =', sq_err)

    # 평균 제곱 오차(mean squared error)
    mse = np.mean(sq_err)
    print('MSE =', mse)

    # RMSE(Root Mean Squared Error)
    print('RMSE =', np.sqrt(mse))
