"""
CNN이 사용하는 파라미터(filter W, bias b)의 초깃값과 학습 끝난 후의 값 비교
"""
import matplotlib.pyplot as plt
import numpy as np

from ch07.simple_convnet import SimpleConvNet
from common.layers import Convolution


def show_filters(filters, num_filters, ncols=8):
    nrows = np.ceil(num_filters / ncols)  # subplot의 행(row) 개수
    for i in range(num_filters):  # 필터 개수만큼 반복
        # subplot 설정
        plt.subplot(nrows, ncols, i+1, xticks=[], yticks=[])
        # subplot에 이미지 그리기
        plt.imshow(filters[i, 0], cmap='gray')
    plt.show()


if __name__ == '__main__':
    # Simple CNN 생성
    cnn = SimpleConvNet()
    # 학습시키기 전 파라미터 - 임의의 값으로 초기화된 필터
    before_filters = cnn.params['W1']
    print(before_filters.shape)  # (30, 1, 5, 5)
    # 학습 전 파라미터를 그래프로 출력
    show_filters(before_filters, 30)

    # 학습 끝난 후 파라미터
    # pickle 파일에 저장된 파라미터를 cnn의 필드로 로드
    cnn.load_params('cnn_params.pkl')
    after_filters = cnn.params['W1']
    print(after_filters.shape)
    # 학습 끝난 후 갱신(업데이트) 된 파라미터를 그래프로 출력
    show_filters(after_filters, 30)

    # 학습 끝난 후 갱신된 파라미터를 실제 이미지 파일에 적용
    lena = plt.imread('lena_gray.png')
    print(lena.shape)  # 이미지 파일이 numpy array로 변환됨

    # 이미지 데이터를 Convolution 레이어의 forward 메소드에 전달하기 위해 2차원 배열을 4차원 배열로 변환
    lena = lena.reshape(1, 1, *lena.shape)
    for i in range(16):  # 필터 16개에 대해 반복
        # 필터
        w = cnn.params['W1'][i]  # 갱신된 필터
        b = 0  # 편향 사용하지 않음
        w = w.reshape(1, *w.shape)
        conv = Convolution(w, b)
        out = conv.forward(lena)
        # pyyplot을 사용하기 위해 4차원을 2차원으로 변환
        out = out.reshape(out.shape[2], out.shape[3])
        plt.subplot(4, 4, i+1, xticks=[], yticks=[])
        plt.imshow(out, cmap='binary')
    plt.show()
