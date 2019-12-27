"""
행렬의 내적(dot product)
A, B, ...: 2차원 이상의 ndarray
x, y, ...: 1차원 ndarray
"""
import numpy as np

x = np.array([1, 2])
W = np.array([[3, 4],
              [5, 6]])
print(x.dot(W))
print(W.dot(x))

A = np.arange(1, 7).reshape((2, 3))
print(A)
B = np.arange(1, 7).reshape((3, 2))
print(B)
print(A.dot(B))  # 2x2 행렬
print(B.dot(A))  # 3x3 행렬
# 행렬의 내적(dot product)은 교환법칙이 성립하지 않는다(AB != BA)

# ndarray.shape -> (x, ), (x, y), (x, y, z), ...
x = np.array([1, 2, 3])
print(x)
print(x.shape)  # (3, )

x = x.reshape((3, 1))
print(x)
print(x.shape)

x = x.reshape((1, 3))
print(x)
print(x.shape)
