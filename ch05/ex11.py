"""2층 신경망 테스트"""
import pickle

import numpy as np
import matplotlib.pyplot as plt
from ch05.ex10_twolayer import TwoLayerNetwork
from dataset.mnist import load_mnist

if __name__ == '__main__':
    np.random.seed(106)
    # MNIST 데이터를 로드
    (X_train, Y_train), (X_test, Y_test) = load_mnist(one_hot_label=True)

    # 2층 신경망 생성
    neural_net = TwoLayerNetwork(784, 32, 10)

    epochs = 100  # 100번 반복
    batch_size = 128  # 한번에 학습시키는 입력 데이터 개수
    learning_rate = 0.1

    iter_size = max(X_train.shape[0] // batch_size, 1)
    print(iter_size)

    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []

    for epoch in range(epochs):
        idx = np.random.permutation(len(X_train))
        X_train_random = X_train[idx]
        Y_train_random = Y_train[idx]
        for i in range(iter_size):
            # 처음 batch_size 개수만큼의 데이터를 입력으로 해서 gradient 계산
            x_batch = X_train_random[i * batch_size: (i + 1) * batch_size]
            y_batch = Y_train_random[i * batch_size: (i + 1) * batch_size]

            grads = neural_net.gradient(x_batch, y_batch)

            # 가중치/편향 행렬 수정
            for key in grads.keys():
                neural_net.params[key] -= learning_rate * grads[key]

        # loss를 계산해서 출력
        print(f'Epoch {epoch + 1} / {epochs}')
        train_loss = neural_net.loss(X_train, Y_train)
        test_loss = neural_net.loss(X_test, Y_test)
        print(f'Train-Loss = {train_loss}')
        print(f'Test-Loss  = {test_loss}')
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)

        # accuracy를 계산해서 출력
        train_acc = neural_net.accuracy(X_train, Y_train)
        test_acc = neural_net.accuracy(X_test, Y_test)
        print(f'Train-Acc = {train_acc}')
        print(f'Test-Acc  = {test_acc}')
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

    # 이 과정을 100회(epochs)만큼 반복
    # 반복할 때마다 학습 데이터 세트를 무작위로 섞는(shuffle) 코드를 추가
    # 각 epoch마다 테스트 데이터로 테스트를 해서 accuracy를 계산
    # 100번의 epoch가 끝났을 때, epochs-loss, epochs-accuracy 그래프를 그림

    plt.plot(np.arange(epochs), train_loss_list, label='Train')
    plt.plot(np.arange(epochs), test_loss_list, label='Test')
    plt.title('LOSS')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

    plt.plot(np.arange(epochs), train_acc_list, label='Train')
    plt.plot(np.arange(epochs), test_acc_list, label='Test')
    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()

    # 신경망에서 학습이 모두 끝난 후 파라미터(가중치/편향 행렬)을 파일에 저장
    # pickle 이용
    with open('mnist_param.pkl', mode='wb') as f:
        pickle.dump(neural_net.params, f)

