"""
MNIST 데이터를 사용한 가중치 초깃값과 신경망 성능 비교
"""
import matplotlib.pyplot as plt
import numpy as np

from ch06.ex02_sgd import Sgd
from ch06.ex03_momentum import Momentum
from ch06.ex05_adam import Adam
from common.multi_layer_net import MultiLayerNet
from dataset.mnist import load_mnist

if __name__ == '__main__':
    # 실험 조건 세팅
    weight_init_types = {
        'std=0.01': 0.01,
        'Xavier': 'sigmoid',  # 가중치 초깃값: N(0, sqrt(1/n))
        'He': 'relu'  # 가중치 초깃값: N(0, sqrt(2/n))
    }

    # 각 실험조건 별로 테스트할 신경망을 생성
    neural_nets = dict()
    train_losses = dict()
    for key, type in weight_init_types.items():
        neural_nets[key] = MultiLayerNet(input_size=784,
                                         hidden_size_list=[100, 100, 100, 100],
                                         output_size=10,
                                         weight_init_std=type,
                                         activation='sigmoid')
        train_losses[key] = []  # 빈 리스트 생성 - 실험(학습)하면서 손실값을 저장

    # MNIST train/test 데이터 로드
    (X_train, Y_train), (X_test, Y_test) = load_mnist(one_hot_label=True)

    iterations = 2_000  # 학습 회수
    batch_size = 128  # 1회 학습에 사용할 샘플 개수(미니배치)
    # optimizer = Sgd(learning_rate=0.01)  # 파라미터 최적화 알고리즘
    # optimizer를 변경하면서 테스트(나중에)
    optimizer = Adam(lr=0.01)
    # optimizer = Momentum(lr=0.01)

    np.random.seed(109)
    # 2,000번 반복하면서
    for _ in range(iterations):
        # 미니배치 샘플 랜덤 추출
        idx = np.random.choice(len(X_train), batch_size)
        X_batch = X_train[idx]
        Y_batch = Y_train[idx]
        # 테스트 신경망 종류마다 반복
        for key, net in neural_nets.items():
            # gradient 계산
            grad = net.gradient(X_batch, Y_batch)
            # 파라미터(W, b) 업데이트
            optimizer.update(net.params, grad)
            # 손실(loss) 계산 -> 리스트에 추가
            train_losses[key].append(net.loss(X_batch, Y_batch))
        # 손실 일부 출력
        if (_ + 1) % 100 == 0:
            print(f'========== iteration {_ + 1} / {iterations} ==========')
            for key in weight_init_types:
                print(key, ':', train_losses[key][-1])

    # x축 - 반복 회수, y축 - 손실 그래프
    x = np.arange(iterations)
    for key in weight_init_types:
        plt.plot(x, train_losses[key], label=key)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
