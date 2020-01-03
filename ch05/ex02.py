"""
f(x, y, z) = (x + y) * z
x = -2, y = 5, z = -4에서의 df/dx, df/dy, df/dz의 값을 ex01에서 구현한 MultiplyLayer 와 AddLayer 클래스를 이용해서 구하라
numerical_gradient 함수에서 계산된 결과와 비교
"""
from ch05.ex01_basic_layer import AddLayer, MultiplyLayer

if __name__ == '__main__':
    x = -2
    y = 5
    z = -4
    print(f'x = {x}, y = {y}, z = {z}')

    xy_layer = AddLayer()
    t = xy_layer.forward(x, y)
    print('x + y =', t)

    xyz_layer = MultiplyLayer()
    f = xyz_layer.forward(t, z)
    print('(x + y) * z =', f)

    delta = 1.0
    dt, dz = xyz_layer.backward(delta)
    print(f'df/d(x + y) = {dt}, df/dz = {dz}')

    dx, dy = xy_layer.backward(dt)
    print(f'df/dx = {dx}, df/dy = {dy}')


    def f(x, y, z):
        return (x + y) * z


    h = 1e-4
    dx = (f(-2 + h, 5, -4) - f(-2 - h, 5, -4)) / (2 * h)
    print('df/dx =', dx)
    dy = (f(-2, 5 + h, -4) - f(-2, 5 - h, -4)) / (2 * h)
    print('df/dy =', dy)
    dx = (f(-2, 5, -4 + h) - f(-2, 5, -4 - h)) / (2 * h)
    print('df/dz =', dz)
