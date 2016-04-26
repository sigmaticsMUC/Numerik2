import numpy as np
from matplotlib import pyplot as mt
# author: Knoll Alexander


def explicit_middle(interval, f, h, y_0):
    """
    Computes numerical solution to f using an explicit midpoint approach.

    Input
    -----
    :param interval: list of equidistant values.
    :param f: function describing the problem. f(t, y) = y'.
    :param h: step size.
    :param y_0: initial value at y(0).

    Return
    ------
    :return ys: list containing values of solution.

    """
    length = len(interval)
    ys = [None for _ in range(length)]

    # set initial value
    ys[0] = y_0
    # provide further information so the numeric method can be executed
    # using explicit euler to generate one more value.
    ys[1] = y_0 + h * f(interval[0], y_0)

    for i in range(1, length-1, 1):
        ys[i+1] = ys[i-1] + 2*h*f(interval[i], ys[i])

    return ys


def implicit_bdf(interval, f, h, y_0):
    """
    Computes numerical solution to f using an implicit bdf approach.
    The equation is solved by using a secant algorithm.

    Input
    -----
    :param interval: list of equidistant values.
    :param f: function describing the problem. f(t, y) = y'.
    :param h: step size.
    :param y_0: initial value at y(0).

    Return
    ------
    :return ys: list containing values of solution.

    """
    length = len(interval)
    ys = [None for _ in range(length)]

    # set initial value
    ys[0] = y_0
    # provide further information so the numeric method can be executed
    # using explicit euler to generate one more value.
    ys[1] = y_0 + h * f(interval[0], y_0)

    for i in range(1, length-1, 1):
        U = (ys[i-1], ys[i])
        func2 = lambda u: u - 4/3 * ys[i] + 1/3 * ys[i-1] - h * 2/3 * f(interval[i+1], u)
        ys[i+1] = U[1] - (U[1]-U[0])/(func2(U[1]) - func2(U[0]))*func2(U[1])
    return ys


def main():
    """
    Defines a problem of the form y' = f(t, y), also an estimated solution at y(10) is necessary.
    Afterwards using an implicite bdf an an explicit midpoint approach to compute approximations for different
    step sizes h.

    The resulting error values will be printed to the console.
    """
    l = 10
    h = 0.1
    y_0 = 1

    #func = lambda t, u:  u*np.cos(t)**3 + u*np.sin(t) + 1/u+1
    func = lambda t, u: -2 * u  + 1
    exact_solution = 0.5000000010305768
    h_list = (1, 10**-1, 10**-2, 10**-3, 10**-4, 10**-5)

    length = len(h_list)
    implicit_bdf_error = [None for _ in range(length)]
    explicit_middle_error = [None for _ in range(length)]

    for i in range(length):
        interval = np.arange(0, l+h_list[i], h_list[i])
        interval_length = len(interval)
        implicit_bdf_error[i] = exact_solution - implicit_bdf(interval, func, h_list[i], y_0)[interval_length-1]
        explicit_middle_error[i] = exact_solution - explicit_middle(interval, func, h_list[i], y_0)[interval_length-1]

    print(implicit_bdf_error)
    print(explicit_middle_error)

    #interval = np.arange(0, l+h, h)
    #res = explicit_middle(interval, func, h, y_0)
    #mt.figure()
    #mt.plot(interval, res, color='red')
    #mt.show()

main()

