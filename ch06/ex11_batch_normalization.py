"""
배치 정규화(Batch Normalization):
신경망의 각 층에 미니배치(mini-batch)를 전달할 때마다 정규화(normalization)를 실행하도록 강제하는 방법
-> 학습속도 개선 - p.213 그림 6-18
-> 파라미터(W, b)의 초깃값에 크게 의존하지 않음 - p.214 그림 6-19
-> 과적합(overfitting)을 억제

y = gamma * x + beta
gamma 파라미터: 정규화된 미니배치를 scale-up/down
beta 파라미터: 정규화된 미니배치를 이동(bias)
배치 정규화를 사용할 때는 gamma와 beta의 초깃값을 설정하고 학습시키면서 계속 갱신(업데이트)함
"""
import matplotlib.pyplot as plt
import numpy as np

from ch06.ex02_sgd import Sgd
from common.multi_layer_net_extend import MultiLayerNetExtend
from dataset.mnist import load_mnist

if __name__ == '__main__':
    np.random.seed(110)
    # p.213 그림 6-18을 그리시오
    # Batch Normalization을 사용하는 신경망과 사용하지 않는 신경망의 학습 속도 비교

    # 배치 정규화를 사용하는 신경망
    bn_neural_net = MultiLayerNetExtend(input_size=784,
                                        hidden_size_list=[100, 100, 100, 100, 100],
                                        output_size=10,
                                        weight_init_std=0.01,
                                        use_batchnorm=True)
    # 배치 정규화를 사용하지 않는 신경망
    neural_net = MultiLayerNetExtend(input_size=784,
                                     hidden_size_list=[100, 100, 100, 100, 100],
                                     output_size=10,
                                     weight_init_std=0.01,
                                     use_batchnorm=False)

    # MNIST 데이터 로드
    (X_train, Y_train), (X_test, Y_test) = load_mnist()
    # optimizer 로드
    optimizer = Sgd(learning_rate=0.01)

    # 미니배치를 20번 학습시키면서 두 신경망에서 정확도(accuracy)를 기록
    iterations = 200
    train_size = 1000
    batch_size = 128
    bn_acc = []
    acc = []

    for _ in range(iterations):
        idx = np.random.choice(train_size, batch_size)
        X_batch = X_train[idx]
        Y_batch = Y_train[idx]

        # 배치 정규화를 사용하지 않는 신경망 학습
        grads = neural_net.gradient(X_batch, Y_batch)
        optimizer.update(neural_net.params, grads)
        a = neural_net.accuracy(X_train[np.arange(train_size)], Y_train[np.arange(train_size)])
        acc.append(a)

        # 배치 정규화를 사용하는 신경망 학습
        bn_grads = bn_neural_net.gradient(X_batch, Y_batch)
        optimizer.update(bn_neural_net.params, bn_grads)
        bn_a = bn_neural_net.accuracy(X_train[np.arange(train_size)], Y_train[np.arange(train_size)])
        bn_acc.append(bn_a)

        print(f'===== iteration {_+1} / {iterations} =====')
        print('BatchNorm:', bn_acc[-1])
        print('Normal:', acc[-1])

    # -> 그래프
    x = np.arange(iterations)
    plt.plot(x, bn_acc, label='Batch Normalization')
    plt.plot(x, acc, '--', label='Normal')
    plt.ylim(0, 1)
    plt.legend()
    plt.show()
