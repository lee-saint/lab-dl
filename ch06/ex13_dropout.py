import matplotlib.pyplot as plt
import numpy as np

from common.multi_layer_net_extend import MultiLayerNetExtend
from common.optimizer import SGD
from dataset.mnist import load_mnist

if __name__ == '__main__':
    np.random.seed(110)
    x = np.random.rand(20)
    print(x)
    mask = x > 0.5
    print(mask)
    print(x * mask)

    # x = np.random.rand(100)
    # plt.hist(x, bins=10, range=(0, 1))
    # plt.show()

    np.random.seed(110)
    # 데이터 준비
    (X_train, Y_train), (X_test, Y_test) = load_mnist(one_hot_label=True)
    # 신경망 생성
    neural_net = MultiLayerNetExtend(input_size=784,
                                     hidden_size_list=[100, 100, 100, 100, 100],
                                     output_size=10,
                                     use_dropout=True,
                                     dropout_ration=0.2)
    optimizer = SGD(lr=0.01)

    X_train = X_train[:500]
    Y_train = Y_train[:500]
    X_test = X_test[:500]  # 실험시간 줄이기 위해서
    Y_test = Y_test[:500]

    epochs = 200  # 1 에포크: 모든 학습 데이터가 1번씩 학습된 경우
    mini_batch_size = 100  # 1번 forward에 보낼 데이터 샘플 개수
    # 학습하면서 학습/테스트 데이터의 정확도를 각 에포크마다 기록
    iterations = len(X_train) // mini_batch_size
    train_acc_list = []
    test_acc_list = []

    for epoch in range(epochs):
        idx = np.random.permutation(len(X_train))
        for iter in range(iterations):
            x_batch = X_train[idx[iter * mini_batch_size: (iter + 1) * mini_batch_size]]
            y_batch = Y_train[idx[iter * mini_batch_size: (iter + 1) * mini_batch_size]]
            grad = neural_net.gradient(x_batch, y_batch)
            optimizer.update(neural_net.params, grad)
        train_acc = neural_net.accuracy(X_train, Y_train)
        train_acc_list.append(train_acc)
        test_acc = neural_net.accuracy(X_test, Y_test)
        test_acc_list.append(test_acc)
        if (epoch + 1) % 10 == 0:
            print(f'epoch {epoch + 1} / {epochs}: train accuracy = {train_acc}, test accuracy = {test_acc}')

    x = np.arange(epochs)
    plt.plot(x, train_acc_list, label='Train Accuracy')
    plt.plot(x, test_acc_list, label='Test Accuracy')
    plt.ylim(0, 1)
    plt.legend()
    plt.show()
