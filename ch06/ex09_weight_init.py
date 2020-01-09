"""
6.2 가중치 초깃값 in Y = X @ W + b
신경망의 파라미터인 가중치 행렬(W)을 처음에 어떻게 초기화하느냐에 따라 신경망의 학습 성능이 달라질 수 있믐
Weight의 초깃값을 모두 0 (또는 모두 균일한 값)으로 하면 학습이 이루어지지 않음
그래서 Weight의 초깃값은 보통 정규분포를 따르는 난수를 랜덤하게 추출해서 생성
But 정규분포의 표준편차에 따라 학습의 성능이 달라짐
1) Weight 행렬의 초깃값을 N(0, 1) 분포를 따르는 난수로 생성하면 활성화 값이 0과 1 주위에 치우쳐 분포
-> 역전파의 gradient 값이 점점 작아지다가 사라지는 현상 발생
-> 기울기 소실(gradient vanishing)
2) Weight 행렬의 초깃값을 N(0, 0.01) 분포를 따르는 난수로 생성하면 활성화 값이 0.5 부근에 집중됨
-> 뉴런 1개짜리 신경망과 다를 바가 없음 (뉴런을 많이 만들 이유가????)
-> 표현력(representational power) 제한
3) Xavier 초깃값: 이전 계층의 노드(뉴런) 개수가 n개이면 N(0, sqrt(1/n))인 분포를 따르는 난수로 생성하는 것
-> 활성화 함수가 sigmoid 또는 tanh인 경우에 좋음
4) He 초깃값: 이전 계층의 노드(뉴런) 개수가 n개이면 N(0, sqrt(2/n))인 분포를 따르는 난수로 생성하는 것
-> 활성화 함수가 ReLU인 경우에 적당
"""
import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def relu(x):
    return np.maximum(x, 0)


if __name__ == '__main__':
    # 은닉층에서 자주 사용하는 3가지 활성화 함수 그래프
    x = np.linspace(-10, 10, 1000)
    y_sig = sigmoid(x)
    y_tanh = tanh(x)
    y_relu = relu(x)
    plt.plot(x, y_sig, label='Sigmoid')
    plt.plot(x, y_tanh, label='Hyperbolic Tangent')
    plt.plot(x, y_relu, label='ReLU')
    plt.legend()
    plt.title('Activation Functions')
    plt.ylim((-1.5, 1.5))
    plt.axvline(color='0.9')
    plt.axhline(color='0.9')
    plt.show()

    # 가상의 신경망에서 사용할 테스트 데이터
    np.random.seed(108)
    x = np.random.randn(1000, 100)  # 정규화가 된 테스트데이터

    node_num = 100  # 은닉층의 노드(뉴런) 개수
    hidden_layer_size = 5  # 은닉층의 개수

    weight_init_types = {
        'std=0.01': 0.01,
        'Xavier': np.sqrt(1/node_num),
        'He': np.sqrt(2/node_num)
    }
    weight_init_results = dict()
    input_data = np.random.randn(1_000, 100)
    for k, v in weight_init_types.items():
        x = input_data
        activations = dict()  # 데이터가 은닉층을 지날 때 출력되는 값을 저장

        for i in range(hidden_layer_size):
            # 은닉층에서 사용하는 가중치 행렬:
            # 평균 0, 표준편차 1인 정규분포(N(0, 1))를 따르는 난수로 가중치 행렬 생성
            # w = np.random.randn(node_num, node_num)
            # w = np.random.randn(node_num, node_num) * 0.01  # N(0, 0.01)
            # w = np.random.randn(node_num, node_num) / np.sqrt(node_num)  # N(0, sqrt(1/n))
            # w = np.random.randn(node_num, node_num) * np.sqrt(2 / node_num)  # N(0, sqrt(2/n))
            w = np.random.randn(node_num, node_num) * v
            # a = x dot w
            a = x.dot(w)  # a = x @ w
            # x = sigmoid(a)  # 활성화 함수 적용 -> 은닉층의 출력(output)
            # x = tanh(a)
            x = relu(a)
            activations[i] = x  # 그래프 그리기 위해 출력 결과 저장

        weight_init_results[k] = activations

    i = 1
    for weight, activation in weight_init_results.items():
        for j, a in activation.items():
            plt.subplot(len(weight_init_results), len(activations), i)
            # subplot(nrows, ncols, index)
            plt.title(f"{weight}, {j+1}-layer")
            plt.hist(a.flatten(), 30, range=(-1, 1))
            i += 1
    plt.show()
