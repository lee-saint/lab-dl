import numpy as np


def and_gate(x):
    # x는 [0, 0], [0, 1], [1, 0], [1, 1] 중 하나인 numpy.ndarray 타입
    # w = [w1, w2]인 numpy.ndarray 가중치와 bias b를 찾음
    w = np.array([0.5, 0.5])
    b = -0.5
    y = np.sum(x * w) + b
    if y > 0:
        return 1
    else:
        return 0


def nand_gate(x):
    w = np.array([-0.5, -0.5])
    b = 0.7
    y = np.sum(x * w) + b
    if y > 0:
        return 1
    else:
        return 0


def or_gate(x):
    w = np.array([0.5, 0.5])
    b = -0.3
    y = np.sum(x * w) + b
    if y > 0:
        return 1
    else:
        return 0


if __name__ == '__main__':
    for x1 in (0, 1):
        for x2 in (0, 1):
            x = np.array([x1, x2])
            print(f'AND({x1}, {x2}) -> {and_gate(x)}')

    for x1 in (0, 1):
        for x2 in (0, 1):
            x = np.array([x1, x2])
            print(f'NAND({x1}, {x2}) -> {nand_gate(x)}')

    for x1 in (0, 1):
        for x2 in (0, 1):
            x = np.array([x1, x2])
            print(f'OR({x1}, {x2}) -> {or_gate(x)}')

