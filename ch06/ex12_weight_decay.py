"""
과적합(overfitting): 모델이 학습 데이터는 정확하게 예측하지만, 학습되지 않은 데이터에 대해 정확도가 떨어지는 현상
overfitting이 나타나는 경우:
    1) 학습 데이터가 적은 경우
    2) 파라미터가 너무 많아 표현력(representational power)이 너무 높은 모델
overfitting 되지 않도록 학습
    1) regularization (정칙화, 정규화, ...): L1, L2-regularization
        손실(비용) 함수에 L1 규제(W) 또는 L2 규제(W**2)를 더해서 파라미터(W, b)를 갱신(업데이트)할 때 파라미터가 더 큰 감소를 하도록 만드는 것
        L + (1/2) * lambda * ||W||**2 : L2 규제
        -> W = W - lr * (dL/dW + lambda * W)
        -> 가중치가 더 큰 값이 더 큰 감소를 일으킴
        L + lambda * ||W|| : L1 규제
        -> W = W - lr * (dL/dW + lambda)
        -> 모든 가중치가 일정한 크기로 감소됨
    2) Dropout: 학습 중에 은닉층의 뉴런을 랜덤하게 골라서 삭제하고 학습시키는 방법. 테스트 시에는 모든 뉴런 사용

    overfitting을 줄이는 전략은 학습 시의 정확도를 일부러 줄이는 것임!!
    -> 학습 데이터의 정확도와 테스트 데이터의 정확도 간의 차이를 줄임
"""
import matplotlib.pyplot as plt
import numpy as np

from common.multi_layer_net import MultiLayerNet
from common.optimizer import SGD
from dataset.mnist import load_mnist

if __name__ == '__main__':
    np.random.seed(110)
    # 데이터 준비
    (X_train, Y_train), (X_test, Y_test) = load_mnist(one_hot_label=True)
    # 신경망 생성
    neural_net = MultiLayerNet(input_size=784,
                               hidden_size_list=[100, 100, 100, 100, 100],
                               output_size=10,
                               weight_decay_lambda=0.05)
    optimizer = SGD()

    # 학습 데이터 개수를 300개로 제한 -> overfitting 만들기 위해서
    X_train = X_train[:300]
    Y_train = Y_train[:300]
    X_test = X_test[:300]  # 실험시간 줄이기 위해서
    Y_test = Y_test[:300]

    epochs = 200  # 1 에포크: 모든 학습 데이터가 1번씩 학습된 경우
    mini_batch_size = 100  # 1번 forward에 보낼 데이터 샘플 개수
    # 학습하면서 학습/테스트 데이터의 정확도를 각 에포크마다 기록
    iterations = len(X_train) // mini_batch_size
    train_acc_list = []
    test_acc_list = []

    for epoch in range(epochs):
        idx = np.random.permutation(len(X_train))
        for iter in range(iterations):
            x_batch = X_train[idx[iter*mini_batch_size: (iter+1)*mini_batch_size]]
            y_batch = Y_train[idx[iter*mini_batch_size: (iter+1)*mini_batch_size]]
            grad = neural_net.gradient(x_batch, y_batch)
            optimizer.update(neural_net.params, grad)
        train_acc = neural_net.accuracy(X_train, Y_train)
        train_acc_list.append(train_acc)
        test_acc = neural_net.accuracy(X_test, Y_test)
        test_acc_list.append(test_acc)
        if (epoch + 1) % 10 == 0:
            print(f'===== epoch {epoch + 1} / {epochs} =====')
            print(f'train accuracy = {train_acc}, test accuracy = {test_acc}')

    x = np.arange(epochs)
    plt.plot(x, train_acc_list, label='Train Accuracy')
    plt.plot(x, test_acc_list, label='Test Accuracy')
    plt.legend()
    plt.show()



