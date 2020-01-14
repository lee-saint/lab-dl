import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from dataset.mnist import load_mnist


def pooling1d(x, pool_size, stride=1):
    n, = x.shape  # n = len(x)
    result_size = (n - pool_size) // stride + 1
    result = np.zeros(result_size)
    for i in range(result_size):
        x_sub = x[i*stride:i*stride+pool_size]
        result[i] = np.max(x_sub)
    return result.astype('i')


def pooling2d(x, pool_h, pool_w, stride=1):
    """

    :param x: 2-dim ndarray
    :param pool_h: pooling window height
    :param pool_w: pooling window width
    :param stride: 보폭
    :return: max-pooling
    """
    xh, xw = x.shape
    oh = (xh - pool_h) // stride + 1
    ow = (xw - pool_w) // stride + 1
    result = [[np.max(x[i*stride:i*stride+pool_h, j*stride:j*stride+pool_w]) for j in range(ow)] for i in range(oh)]
    return np.array(result)


if __name__ == '__main__':
    np.random.seed(114)
    x = np.random.randint(10, size=10)
    print(x)

    pooled = pooling1d(x, pool_size=2, stride=2)
    print(pooled)

    pooled = pooling1d(x, pool_size=4, stride=2)
    print(pooled)

    pooled = pooling1d(x, pool_size=4, stride=3)
    print(pooled)

    # pooling2d 테스트
    x = np.random.randint(100, size=(8, 8))
    print(x)

    pooled = pooling2d(x, 4, 4, 4)
    print(pooled)

    print()
    x = np.random.randint(100, size=(5, 5))
    print(x)
    pooled = pooling2d(x, pool_h=3, pool_w=3, stride=2)
    print(pooled)

    # MNIST 데이터셋 로드
    # 손글씨 이미지 하나 선택: shape=(1, 28, 28)
    # 선택된 이미지를 pyplot을 이용하여 출력
    # window shape=(4, 4), stride=4 pooling -> output shape=(7,7)
    # pyplot으로 출력
    (x_train, y_train), (x_test, y_test) = load_mnist(flatten=False)
    img = np.array(x_train[1728]).reshape((28, 28))
    plt.imshow(img, cmap='binary')
    plt.show()

    pooled_img = pooling2d(img, pool_h=4, pool_w=4, stride=4)
    print(pooled_img.shape)
    plt.imshow(pooled_img, cmap='binary')
    plt.show()

    # 이미지 파일을 오픈: (height, width, color)
    # Red, Green, Blue에 해당하는 2차원 배열을 추출
    # 각각의 2차원 배열을 window shape=(16, 16), stride=16으로 pooling
    # pooling된 결과(shape)를 확인, pyplot

    img = Image.open('sample2.jpg')
    img = np.array(img)
    img_r = img[:, :, 0]
    img_g = img[:, :, 1]
    img_b = img[:, :, 2]
    pooled_img_r = pooling2d(img_r, pool_h=16, pool_w=16, stride=16)
    pooled_img_g = pooling2d(img_g, pool_h=16, pool_w=16, stride=16)
    pooled_img_b = pooling2d(img_b, pool_h=16, pool_w=16, stride=16)
    print(pooled_img_r.shape)
    # plt.imshow(pooled_img_r)
    # plt.show()
    # img_rg = np.append(pooled_img_r, pooled_img_g, axis=2)
    # img_rgb = np.append(img_rg, pooled_img_b, axis=2)
    img_rgb = np.array([pooled_img_r, pooled_img_g, pooled_img_b])
    img_rgb = np.moveaxis(img_rgb, 0, 2)
    plt.imshow(img_rgb)
    plt.show()

