"""
교재 p.160 그림 5-15의 빈칸 채우기
apple = 100원, n_a = 2개
orange = 150원, n_o = 3개
tax = 1.1
이라고 할 때, 전체 과일 구매 금액을 AddLayer와 MultiplyLayer를 사용해서 계산하시오.
df/dapple, df/dn_a, df/dorange, df/dn_o, df/dtax 값도 각각 계산하시오.
"""
from ch05.ex01_basic_layer import MultiplyLayer, AddLayer

if __name__ == '__main__':
    apple = 100
    n_a = 2
    apple_layer = MultiplyLayer()
    apple_price = apple_layer.forward(apple, n_a)

    orange = 150
    n_o = 3
    orange_layer = MultiplyLayer()
    orange_price = orange_layer.forward(orange, n_o)

    fruit_layer = AddLayer()
    fruit_price = fruit_layer.forward(apple_price, orange_price)

    tax = 1.1
    tax_layer = MultiplyLayer()
    total_price = tax_layer.forward(fruit_price, tax)
    print('total_price =', total_price)

    delta = 1.0
    d_fruit_price, d_tax = tax_layer.backward(delta)
    d_apple_price, d_orange_price = fruit_layer.backward(d_fruit_price)
    d_apple, d_n_a = apple_layer.backward(d_apple_price)
    d_orange, d_n_o = orange_layer.backward(d_orange_price)

    print('df/d_apple =', d_apple)
    print('df/d_n_a =', d_n_a)
    print('df/d_orange =', d_orange)
    print('df/d_n_o =', d_n_o)
    print('df/d_tax =', d_tax)

