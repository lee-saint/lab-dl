class MultiplyLayer:
    def __init__(self):
        # forward 메소드가 호출될 때 전달되는 입력값을 저장하기 위한 변수 -> backward 메소드가 호출될 때 사용하는 값들
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        return x * y

    def backward(self, delta_out):
        dx = delta_out * self.y
        dy = delta_out * self.x
        return dx, dy


class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        return x + y

    def backward(self, delta_out):
        dx = delta_out
        dy = delta_out
        return dx, dy


if __name__ == '__main__':
    apple_layer = MultiplyLayer()  # MultiplyLayer 객체 생성

    apple = 100  # 사과 한 개의 가격: 100원
    n = 2  # 사과 개수: 2개
    apple_price = apple_layer.forward(apple, n)  # 순방향 전파(forward propagation)
    print('사과 2개 가격:', apple_price)

    # tax_layer를 MultiplyLayer 객체로 생성
    tax_layer = MultiplyLayer()

    # tax = 1.1로 설정해서 사과 2개 구매할 때 총 금액을 계산
    tax = 1.1
    total_price = tax_layer.forward(apple_price, tax)
    print('세금 포함 가격:', total_price)

    # f = a * n * t 라고 할 때
    # 사과 개수가 1 증가하면 전체 가격은 얼마 증가? -> df/da
    # 사과 가격이 1 증가하면 전체 가격은 얼마 증가? -> df/dn
    # tax가 1 증가하면 전체 가격은 얼마 증가? -> df/dt

    # backward propagation(역전파)
    delta = 1.0
    d_price, d_tax = tax_layer.backward(delta)
    print('d_price =', d_price)
    print('d_tax =', d_tax)  # df/dt: tax 변화율에 대한 전체 가격 변화율

    d_apple, d_n = apple_layer.backward(d_price)
    print('d_apple =', d_apple)  # df/da
    print('d_n =', d_n)  # df/dn

    # AddLayer test
    add_layer = AddLayer()
    x = 100
    y = 200
    f = add_layer.forward(x, y)
    print('f =', f)
    delta = 1.5
    dx, dy = add_layer.backward(delta)
    print('dx =', dx)
    print('dy =', dy)

