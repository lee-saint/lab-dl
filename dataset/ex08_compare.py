"""
파라미터 최적화 알고리즘 6개의 성능 비교 - 손실(loss), 정확도(accuracy)
"""
import matplotlib.pyplot as plt
import numpy as np

from ch05.ex10_twolayer import TwoLayerNetwork
from ch06.ex02_sgd import Sgd
from ch06.ex03_momentum import Momentum
from ch06.ex05_adam import Adam
from ch06.ex06_rmsprop import RMSProp
from ch06.ex07_nesterov import Nesterov
from dataset.mnist import load_mnist
from ex04_adagrad import AdaGrad

if __name__ == '__main__':
    # MNIST(손글씨 이미지) 데이터 로드
    (X_train, Y_train), (X_test, Y_test) = load_mnist(one_hot_label=True)

    # 최적화 알고리즘을 구현한 클래스의 인스턴스를 dict에 저장
    optimizers = dict()
    optimizers['SGD'] = Sgd()
    optimizers['momentum'] = Momentum()
    optimizers['AdaGrad'] = AdaGrad()
    optimizers['Adam'] = Adam()
    optimizers['RMSProp'] = RMSProp()
    optimizers['Nesterov'] = Nesterov()
    # 나머지 알고리즘 구현 객체 저장

    # 은닉층 1개, 출력층 1개로 이루어진 신경망을 optimizers 개수만큼 생성
    # 각 신경망에서 손실을 저장할 dict 생성
    neural_nets = dict()
    train_losses = dict()
    for key in optimizers:
        neural_nets[key] = TwoLayerNetwork(784, 32, 10)
        train_losses[key] = []  # loss의 이력(history)을 저장

    # 각각의 신경망을 학습시키면서 loss를 계산/기록
    iterations = 2_000  # 총 학스 회수
    batch_size = 128    # 한 번 학습에서 사용할 미니 배치 크기
    train_size = X_train.shape[0]
    np.random.seed(108)
    for i in range(iterations):  # 2,000번 학습 반복
        # 학습 데이터(X_train), 학습 레이블(Y_train)에서 미니 배치 크기만큼 랜덤하게 데이터를 선택
        batch_mask = np.random.choice(train_size, batch_size)
        # 0 ~ 59,999 사이의 숫자(train_size) 중에서 128(batch_size)개의 숫자를 임의로 선택
        # 학습에 사용할 미니 배치 데이터/레이블 선택
        X_batch = X_train[batch_mask]
        Y_batch = Y_train[batch_mask]
        # 선택된 학습 데이터/레이블을 사용해 gradient를 계싼
        for key in optimizers:
            # 각각의 최적화 알고리즘에 대해 gradient 계산
            gradients = neural_nets[key].gradient(X_batch, Y_batch)
            # 각 최적화 알고리즘의 파라미터 업데이트 기능을 사용
            optimizers[key].update(neural_nets[key].params, gradients)
            # 미니 배치의 손실을 계산
            loss = neural_nets[key].loss(X_batch, Y_batch)
            train_losses[key].append(loss)
        # 100번째 학습마다 계산된 손실을 출력
        if (i + 1) % 100 == 0:
            print(f'=========== training #{i + 1} ==========')
            for key in optimizers:
                print(key, ':', train_losses[key][-1])

    print()
    for key in optimizers:
        print(key, ':', neural_nets[key].accuracy(X_test, Y_test))
        plt.plot(np.arange(iterations), train_losses[key], label=key)
    plt.title('Losses')
    plt.xlabel('# of training')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
