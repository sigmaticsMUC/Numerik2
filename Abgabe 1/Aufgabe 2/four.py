import numpy as np
from matplotlib import pyplot as mt


def explicit_euler(interval, h, y_0):

    length = len(interval)
    ys = [None for _ in range(length)]
    ys[0] = y_0
    for i in range(length-1):
        ys[i+1] = (1-2*h)*ys[i]+h

    return ys


def implicit_euler(interval, h, y_0):

    length = len(interval)
    ys = [None for _ in range(length)]
    ys[0] = y_0
    for i in range(length-1):
        ys[i+1] = ((ys[i] + h)/(1+2*h))

    return ys


def adams_moulton2(interval, h, y_0):

    length = len(interval)
    ys = [None for _ in range(length)]
    ys[0] = y_0
    for i in range(length-1):
        ys[i+1] = (((1-h)*ys[i] + h)/(1+h))
    return ys


def main():

    l = 2.5
    h = 0.1
    h2 = 0.001
    epsilon = 0.0001
    y_0 = 1.0
    interval = np.arange(0, l+h, h)
    interval2 = np.arange(0, l, h2)

    explicit_euler_ys = explicit_euler(interval, h, y_0)
    exact_ys = list(map(lambda t: (0.5+epsilon)*np.e**(-2*t) + 0.5, interval2))
    implicit_euler_ys = implicit_euler(interval, h, y_0)
    moulton_ys = adams_moulton2(interval, h, y_0)

    mt.figure()
    mt.xlabel('t')
    mt.ylabel('y(t)')
    mt.plot(interval, explicit_euler_ys, color='red')
    mt.plot(interval2, exact_ys, color="black")
    mt.plot(interval, implicit_euler_ys, color="blue")
    mt.plot(interval, moulton_ys, color="green")
    mt.title("Comparison of numeric methods using $h = 0,1$")
    mt.legend(('explicit euler', 'analytic solution', 'implicit euler', 'adam moulton (order 2)'), loc='upper center')

    mt.show()


main()
