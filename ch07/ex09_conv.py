"""
im2col 함수를 사용한 Convolution 구현
"""
import numpy as np
from common.util import im2col

if __name__ == '__main__':
    np.random.seed(115)

    # p.238 그림 7-11 참조
    # 가상의 이미지 데이터 1개를 생성
    x = np.random.randint(10, size=(1, 3, 7, 7))
    print(x, ', shape:', x.shape)

    # (3, 5, 5) 크기의 필터 1개 생성
    # (fn, c, fh, fw) = (필터 개수, color-dept, 필터 height, 필터 width)
    w = np.random.randint(5, size=(1, 3, 5, 5))
    print(w, ', shape:', w.shape)
    # 필터를 stride=1, padding=0으로 해서 convolution 연산
    # 필터를 1차원으로 펼침 -> c*fh*fw = 3 * 5 * 5 = 75

    # 이미지 데이터 x를 함수 im2col에 전달
    x_col = im2col(x, filter_h=5, filter_w=5, stride=1, pad=0)
    print('x_col:', x_col.shape)
    # (9, 75) = (oh*ow, c*fh*fw)

    # 4차원 배열인 필터 w를 2차원 배열로 변환
    w_col = w.reshape(1, -1)  # row의 개수가 1, 모든 원소들은 column으로
    print('w_col:', w_col.shape)
    w_col = w_col.T
    print('w_col:', w_col.shape)

    # 2차원으로 변환된 이미지와 필터를 행렬 dot product 연산
    out = x_col.dot(w_col)
    print('out:', out.shape)

    # dot product의 결과를 (fn, oh, ow, ?) 형태로 reshape
    out = out.reshape(1, 3, 3, -1)
    print('out:', out.shape)  # (1, 3, 3, 1) = (fn, oh, ow, c)
    out = out.transpose(0, 3, 1, 2)
    print('out:', out.shape)  # (1, 1, 3, 3)

    # p.238 그림 7-12, p.244 그림 7-19 참조
    # 가상으로 생성한 이미지 데이터 x와 2차원을 변환한 x_col 을 사용
    # (3, 5, 5) 필터를 10개 생성 -> w.shape=(10, 3, 5, 5)
    w = np.random.randint(5, size=(10, 3, 5, 5))

    # w를 변형(reshape): (fn, c*fh*fw)
    w_col = w.reshape((10, -1))
    print('w_col shape:', w_col.shape)

    # x_col @ w_col.T의 shape 확인
    out = x_col.dot(w_col.T)
    print('out shape:', out.shape)

    # dot 연산의 결과를 변형(reshape): (n, oh, ow, fn)
    # reshape된 결과에서 네 번째 축이 두 번째 축이 되도록 전치(transpose)
    out = out.reshape(1, 3, 3, -1).transpose(0, 3, 1, 2)
    print(out.shape)

    # p.239 그림 7-13, p.244 그림 7-19 참조
    # (3, 7, 7) 이미지 12개를 난수로 생성 -> (n, c, h, w) = (12, 3, 7, 7)
    x = np.random.randint(10, size=(12, 3, 7, 7))

    # (3, 5, 5) shape의 필터 10개 난수로 생성 -> (fn, c, h, w) = (10, 3, 5, 5)
    w = np.random.randint(5, size=(10, 3, 5, 5))

    # stride=1, padding=0일 때, output height, output width = ?
    n, c, xh, xw = x.shape
    fn, c, fh, fw = w.shape
    stride = 1
    oh = (xh - fh) // stride + 1  # (h - fh + 2 * p) // s + 1 = (7 - 5 + 2 * 0) // 1 + 1 = 3
    ow = (xw - fw) // stride + 1  # (w - fw + 2 * p) // s + 1 = 3
    print(oh, ow)

    # 이미지 데이터 x를 im2col 함수 사용해서 x_col로 변환 -> shape?
    x_col = im2col(x, 5, 5, stride=1, pad=0)
    print('x_col shape:', x_col.shape)
    # (108, 75) = (n * oh * ow, c * fh * fw)

    # 필터 w를 x_col과 dot 연산을 할 수 있도록 reshape & transpose
    w_col = w.reshape((10, -1)).T
    print('w_col shape:', w_col.shape)  # (75, 10) = (c * fh * fw, fn)

    # x_col @ w_col
    out = x_col.dot(w_col)

    # @ 연산의 결과를 reshape & transpose
    out = out.reshape((12, oh, ow, -1)).transpose(0, 3, 1, 2)
    print('out shape:', out.shape)

