import numpy as np


class Affine:
    def __init__(self, W, b):
        self.W = W  # weight 행렬
        self.b = b  # bias 행렬
        self.X = None  # 입력 행렬을 저장할 field
        self.dW = None  # W 행렬 gradient -> W = W - lr * dW
        self.db = None  # b 행렬 gradient -> b = b - lr * db

    def forward(self, X):
        self.X = X  # 나중에 역전파에서 사용
        out = X.dot(self.W) + self.b
        return out

    def backward(self, dout):
        # b 행렬 방향으로의 gradient
        self.db = np.sum(dout, axis=0)
        # Z 행렬 방향으로의 gradient -> W방향, X방향
        self.dW = self.X.T.dot(dout)  # GD를 사용해서 W, b를 fitting할 때 사용
        dX = dout.dot(self.W.T)
        return dX


if __name__ == '__main__':
    np.random.seed(103)
    X = np.random.randint(10, size=(2, 3))  # 입력 행렬

    W = np.random.randint(10, size=(3, 5))  # 가중치 행렬
    b = np.random.randint(10, size=5)  # bias 행렬

    affine = Affine(W, b)  # Affine 클래스의 객체 생성
    Y = affine.forward(X)  # Affine의 출력값
    print('Y =', Y)

    dout = np.random.randn(2, 5)
    dX = affine.backward(dout)
    print('dX =', dX)
    print('dW =', affine.dW)
    print('db =', affine.db)
