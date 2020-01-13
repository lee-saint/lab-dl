"""
CNN(Convolutional Neural Network, 합성곱 신경망)
원래 convolution 연산은 영상/음성 처리(image/audio processing)에서 신호를 변환하기 위한 연산으로 사용
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.signal import convolve, correlate

if __name__ == '__main__':
    # jpg 파일 오픈
    img = Image.open('sample2.jpg')
    img_pixel = np.array(img)
    print(img_pixel.shape)   # (height, width, color-depth)
    # 머신러닝 라이브러리에 따라 color 표기의 위치가 달라짐
    # Tensorflow: channel-last 방식 / color-depth가 3차원 배열의 마지막 차원
    # Theano: channel-first 방식 / color-depth가 3차원 배열의 첫번째 차원 (c, h, w)
    # Keras: 두 가지 방식 모두 지원

    plt.imshow(img_pixel)
    plt.show()

    # 이미지의 RED 값 정보
    print(img_pixel[:, :, 0])
    # plt.imshow(img_pixel[:, :, 0], cmap='magma')
    # plt.axis('off')
    # plt.xticks([])
    # plt.yticks([])
    # plt.tight_layout()
    # plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
    # plt.savefig("nopadding.png",
    #             bbox_inces='tight',
    #             pad_inches=0,
    #             dpi=100
    #             )
    # plt.show()

    # (3, 3, 3) 필터
    filter = np.zeros((3, 3, 3))
    filter[1, 1, 0] = 1.0
    transformed = correlate(img_pixel, filter, mode='same') / 255
    plt.imshow(transformed)
    plt.show()
