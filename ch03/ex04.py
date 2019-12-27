import numpy as np

from ch03.ex01 import sigmoid

x = np.array([1, 2])

# a1 = x @ W1 + b1
W1 = np.array([[1, 2, 3],
              [4, 5, 6]])
b1 = np.array([1, 2, 3])

a1 = x.dot(W1) + b1
print('a(1) =', a1)

# 출력 a1에 활성화 함수를 sigmoid 함수로 적용
z1 = sigmoid(a1)
print('z(1) =', z1)

# 두번째 은닉층에 대한 가중치 행렬 W2와 bias 행렬 b2를 작성
W2 = np.array([[0.1, 0.4],
               [0.2, 0.5],
               [0.3, 0.6]])
b2 = np.array([0.1, 0.2])
a2 = z1.dot(W2) + b2
print('a(2) =', a2)

# a2에 활성화 함수(sigmoid)를 적용
y = sigmoid(a2)
print('y =', y)
