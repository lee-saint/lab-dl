import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import numpy as np


def fn(x, y):
    """f(x, y) = (1/20) * x**2 + y**2"""
    return x**2 / 20 + y**2


def fn_derivative(x, y):
    """편미분 df/dx, df/dy 튜플을 리턴"""
    return x / 10, 2 * y


if __name__ == '__main__':
    x = np.linspace(-10, 10, 1000)  # x 좌표
    y = np.linspace(-10, 10, 1000)  # y 좌표
    # 3차원 그래프를 그리기 위해서
    X, Y = np.meshgrid(x, y)
    Z = fn(X, Y)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # projection 파라미터를 사용하려면 mpl_toolkits.mplot3d 패키지가 필요
    ax.contour3D(X, Y, Z,
                 100,            # 등고선의 개수
                 cmap='binary')  # 등고선 색상 맵(color map)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    # 등고선(contour) 그래프
    plt.contour(X, Y, Z, 100, cmap='binary')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.show()

