"""
2차원 Convolution(합성곱) 연산
"""
import numpy as np


def convolution_2d(x, w):
    """x, w: 2d ndarray. x.shape >= w.shape
    x와 w의 교차상관 연산 결과를 리턴
    """
    xh, xw = x.shape
    wh, ww = w.shape
    rows = xh - wh + 1
    cols = xw - ww + 1
    # conv = []
    # for i in range(rows):
    #     row = []
    #     for j in range(cols):
    #         row.append(np.sum(x[i:i+wh, j:j+ww] * w))
    #     conv.append(row)
    conv = [[np.sum(x[i:i+wh, j:j+ww] * w) for j in range(cols)] for i in range(rows)]
    return np.array(conv)


if __name__ == '__main__':
    np.random.seed(113)

    x = np.arange(1, 10).reshape((3, 3))
    print(x)
    w = np.array([[2, 0],
                  [0, 0]])
    print(w)

    # 2d 배열 x의 가로(width) xw, 세로(height) xh
    xh, xw = x.shape
    # 2d 배열 w의 가로 ww, 세로 wh
    wh, ww = w.shape

    x_sub1 = x[0:wh, 0:ww]  # x[0:2, 0:2]
    print(x_sub1)
    fma1 = np.sum(x_sub1 * w)
    print(fma1)
    x_sub2 = x[0:wh, 1:1+ww]  # x[0:2, 1:3]
    print(x_sub2)
    fma2 = np.sum(x_sub2 * w)
    x_sub3 = x[1:1+wh, 0:ww]  # x[1:3, 0:2]
    fma3 = np.sum(x_sub3 * w)
    x_sub4 = x[1:1+wh, 1:1+ww]  # x[1:3, 1:3]
    fma4 = np.sum(x_sub4 * w)
    conv = np.array([fma1, fma2, fma3, fma4]).reshape((2, 2))
    print(conv)
    x = np.arange(12).reshape((3, 4))

    conv = convolution_2d(x, w)
    print(conv)

    x = np.random.randint(10, size=(5, 5))
    w = np.random.randint(5, size=(3, 3))
    print(x)
    print(w)
    print(convolution_2d(x, w))
