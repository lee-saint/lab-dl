"""
Simple Convolutional Neural Network(CNN)
p.228 그림 7-2
"""
from collections import OrderedDict
import numpy as np

from common.layers import Convolution, Relu, Pooling, Affine, SoftmaxWithLoss
from common.trainer import Trainer
from dataset.mnist import load_mnist


class SimpleConvNet:
    """
    1st hidden layer: Convolution -> ReLU -> Pooling
    2nd hidden layer: Affine -> ReLU (fully-connected network, 완전연결층)
    출력층: Affine -> SoftmaxWithLoss
    """
    def __init__(self, input_dim=(1, 28, 28), conv_params={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1}, hidden_size=100, output_size=10, weight_init_std=0.01):
        """인스턴스 초기화 - CNN 구성, 변수들 초기화"""
        fn, fs, stride, pad = conv_params['filter_num'], conv_params['filter_size'], conv_params['stride'], conv_params['pad']
        H, O = hidden_size, output_size
        I = input_dim[1]
        conv_output_size = (I - fs + 2 * pad) // stride + 1
        pool_output_size = int(fn * (conv_output_size/2) * (conv_output_size/2))

        # CNN Layer에서 필요한 파라미터
        self.params = dict()
        self.params['W1'] = weight_init_std * np.random.randn(fn, input_dim[0], fs, fs)   # Convolution 계층 파라미터
        self.params['b1'] = np.zeros(fn)
        self.params['W2'] = weight_init_std * np.random.randn(pool_output_size, H)  # Affine 계층 파라미터
        self.params['b2'] = np.zeros(H)
        self.params['W3'] = weight_init_std * np.random.randn(H, O)  # 출력층 파라미터
        self.params['b3'] = np.zeros(O)

        # CNN layer(계층) 생성, 연결
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], stride, pad)
        self.layers['ReLU1'] = Relu()
        self.layers['Pool1'] = Pooling(2, 2, 2)
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['ReLU2'] = Relu()
        self.layers['Affine3'] = Affine(self.params['W3'], self.params['b3'])

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        score = self.predict(x)
        loss = self.last_layer.forward(score, t)
        return loss

    def gradient(self, x, t):
        self.loss(x, t)

        dout = 1
        dout = self.last_layer.backward(dout)
        for layer in reversed(list(self.layers.values())):
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Conv1'].dW
        grads['b1'] = self.layers['Conv1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        grads['W3'] = self.layers['Affine3'].dW
        grads['b3'] = self.layers['Affine3'].db

        return grads

    def accuracy(self, x, t):
        score = self.predict(x)
        pred = np.argmax(score, axis=1)
        acc = np.mean(pred == t)
        return acc


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_mnist(flatten=False)
    conv = SimpleConvNet()
    score = conv.predict(x_train[:5])
    print(score.shape)
    loss = conv.loss(x_train[:5], y_train[:5])
    print(loss)
    acc = conv.accuracy(x_train[:5], y_train[:5])
    print(acc)
    grads = conv.gradient(x_train[:5], y_train[:5])
    for grad in grads:
        print(f'{grad} shape: {grads[grad].shape}')

    trainer = Trainer(conv, x_train[:5000], y_train[:5000], x_test[:1000], y_test[:1000], optimizer='Adam', evaluate_sample_num_per_epoch=1000)
    trainer.train()
