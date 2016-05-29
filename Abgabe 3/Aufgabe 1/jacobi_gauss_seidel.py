from scipy.constants import g
from matplotlib import pyplot as plt
import numpy as np
import numpy.linalg as nplin
import threading as thr
import time

length = 10.0
width = 0.1
depth = 0.05
density = 7850
E = 2e11
I = (width*(depth**3))/12.0


def compute_exact(space):
    """
    Computes exact solution y(x) over given interval.
    :param space: interval.
    :return exact solution.
    """
    return [(f()*(x**2)*(length - x)**2)/(24 * E * I) for x in space]


def f():
    """
    Computes force per unit length which influences the rod.
    :return force influencing rod.
    """
    return density*width*depth*g


def apply_weight_to_f(fs, h):
    """
    Applies extra weight of 500kg between x=3 and x=4 of a list.
    :param fs: force list:
    :param h: step size of fs
    :return fs: force list with extra weights.
    """
    for i in range(len(fs)):
        if 3 <= h*i <= 4:
            fs[i] += 500
            

def create_matrix(n):
    """
    Creates quadratic matrix.
    :param n: dimension of matrix.
    :return matrix: quadratic matrix of dimension n.
    """
    matrix = [[] for _ in range(n)]
    matrix[0] = [12, -6, 4.0/3.0] + [0 for _ in range(n-3)]
    matrix[1] = [-4, 6, -4, 1] + [0 for _ in range(n-4)]
    matrix[n-2] = [0 for _ in range(n-4)] + [1, -4, 6, -4]
    matrix[n-1] = [0 for _ in range(n-3)] + [4.0/3.0, -6, 12]
    for j in range(2, n-2, 1):
        matrix[j] = [0 for _ in range(j-2)] + [1, -4,  6, -4, 1] + [0 for _ in range(n-3-j)]
    return matrix


def inner_sum(ys, index):
    """
    Computes inner sum for the gauss-seidel algorithm.
    :param ys: solution vector (as a list).
    :param index: current working index, used as a reference point to calculate results.
    :return result: inner sum of gauss-seidel algorithm.

    """
    if 1 < index < len(ys)-2:
        result = ys[index-2] - 4 * ys[index-1] - 4 * ys[index+1] + ys[index+2]
    elif index == 0:
        result = -6 * ys[index+1] + (4.0/3.0) * ys[index+2]
    elif index == 1:
        result = -4 * ys[index-1] - 4 * ys[index+1] + ys[index+2]
    elif index == len(ys)-2:
        result = ys[index-2] - 4 * ys[index-1] - 4 * ys[index+1]
    else:
        result = (4.0/3.0) * ys[index-2] - 6 * ys[index-1]
    return result


def seidel(ys, bs):
    """
    Performs one iteration of the gauss-seidel algorithm. Calls inner_sum function.
    Changes state of given solution vector.
    :param ys: current solution vector (as a list). Note that this instance gets manipulated.
    :param bs: right side of the equation (as a list)
    :return ys, ys_old: list of new values and list of old values.
    """
    ys_old = list(ys)
    for i in range(len(ys)):
        diag_elem = 6.0
        if i == 0 or i == len(ys)-1:
            diag_elem = 12.0
        ys[i] = (bs[i] - inner_sum(ys, i))/diag_elem
    return ys, ys_old


def solve(ys, bs, max_iter=40000, tol=1e-6):
    """
    Runs multiple iterations as long as either the tolerance condition, or the max_iter condition, is satisfied.
    Calls seidel on each iteration.
    :param ys: current solution vector (as a list). Note that this instance gets manipulated.
    :param bs: right side of the equation (as a list).
    :param max_iter: number of maximum iterations.
    :param tol: stops iteration if max(x^m+1 - x^m) < tol.
    :return it_counter: number of iterations needed.
    """
    it_counter = 0
    iterating = True
    while iterating:
        _, ys_old = seidel(ys, bs)
        step = max([a_i - b_i for a_i, b_i in zip(ys, ys_old)])
        it_counter += 1
        if it_counter >= max_iter or abs(step) <= tol:
            iterating = False
    return it_counter


def compute_error(numerical, exact):
    """
    Computes error between exact and numerical solution.
    :param numerical: numerical solution as a list
    :param exact: exact solution as a list
    :return max_error, max_error_index: value of max error and its index

    """
    difference = [abs(a_i - b_i) for (a_i, b_i) in zip(exact, numerical)]
    max_error = max(difference)
    max_error_index = difference.index(max_error)
    return max_error, max_error_index


def compute_conditions(k_limit):
    """
    Computes conditions for each matrix up to k_limit. Note that if k is great, the memory necessary also increases.
    :param k_limit: limit for k
    :return cond_list: list of condition values.
    """
    n_list = [10 * (2**k) + 1 for k in range(1, k_limit+1, 1)]
    cond_list = [0 for _ in n_list]
    for n in n_list:
        index = n_list.index(n)
        matrix = create_matrix(n)
        inverse_matrix = nplin.inv(matrix)
        matrix_norm = nplin.norm(matrix)
        inverse_matrix_norm = nplin.norm(inverse_matrix)
        cond_list[index] = matrix_norm*inverse_matrix_norm
    return cond_list


def multi_solve(k_limit, max_iter=10000, tol=1e-10):
    """
    Solves the problem using threads for matrices up to k_limit.
    After each thread was started, the routine waits for each of them to finish before proceeding.
    Plots results afterwards.
    :param k_limit: limit for k
    :param max_iter: number of maximum iterations.
    :param tol: stops iteration if max(x^m+1 - x^m) < tol.
    """
    start = time.time()
    n_list = [10 * (2**k) + 1 for k in range(1, k_limit+1, 1)]
    thread_list = [None for _ in n_list]
    line_space_list = [[] for _ in n_list]
    ys_list = [0 for _ in n_list]
    for n in n_list:
        index = n_list.index(n)
        h = length/(n+1)
        ys_list[index] = [0 for _ in range(n)]
        fs = [f() for _ in range(n)]
        bs = list(map(lambda x: x * ((h**4) / (E*I)), fs))
        line_space_list[index] = np.linspace(0, length, n+2)
        thread_list[index] = thr.Thread(target=solve, args=(ys_list[index], bs, max_iter, tol))
        thread_list[index].start()

    for thread in thread_list:
        index = thread_list.index(thread)
        thread.join()
        ys_list[index] = [0] + ys_list[index] + [0]
        plt.figure('Standard solution')
        plt.plot(line_space_list[index], [-x for x in ys_list[index]])
        error , error_index = compute_error(ys_list[index], compute_exact(line_space_list[index]))
        print("\tMax error: %s for k=%s: at index: %s of %s"%(error, index+1, error_index, len(ys_list[index])-1))
    end = time.time()
    plt.title('Plot for k = 1 to %s with TOL=%s'%(k_limit,tol))
    plt.legend(('k=1', 'k=2','k=3','k=4','k=5','k=6','k=7','k=8','k=9','k=10'), loc='best')
    plt.grid()
    print("\ttime needed in seconds: ", end - start)
    #plt.show()


def multi_solve2(k_limit, max_iter=10000, tol=1e-10):
    """
    Solves the problem using threads for matrices up to k_limit. Applies extra weight to right side
    of the equation b. After each thread was started, the routine waits for each of them to finish before proceeding.
    Plots results afterwards and marks point of maximum bend.
    :param k_limit: limit for k
    :param max_iter: number of maximum iterations.
    :param tol: stops iteration if max(x^m+1 - x^m) < tol.
    """
    start = time.time()
    n_list = [10 * (2**k) + 1 for k in range(1, k_limit+1, 1)]
    thread_list = [None for _ in n_list]
    line_space_list = [[] for _ in n_list]
    ys_list = [0 for _ in n_list]
    for n in n_list:
        index = n_list.index(n)
        h = length/(n+1)
        ys_list[index] = [0 for _ in range(n)]
        fs = [f() for _ in range(n)]
        apply_weight_to_f(fs, h)
        bs = list(map(lambda x: x * ((h**4) / (E*I)), fs))
        line_space_list[index] = np.linspace(0, length, n+2)
        thread_list[index] = thr.Thread(target=solve, args=(ys_list[index], bs, max_iter, tol))
        thread_list[index].start()

    for thread in thread_list:
        index = thread_list.index(thread)
        thread.join()
        ys_list[index] = [0] + ys_list[index] + [0]
        plt.figure('Solution with extra weight')
        plt.plot(line_space_list[index], [-x for x in ys_list[index]])
        max_bend = max(ys_list[index])
        max_bend_pos = list(ys_list[index]).index(max_bend)
        plt.plot(max_bend_pos * (length/(len(ys_list[index])+1)), -max_bend, 'o')
    end = time.time()
    plt.title('Plot for k = 1 to %s with TOL=%s'%(k_limit,tol))
    plt.grid()
    print("\ttime needed in seconds: ", end - start)
    #plt.show()


def main():
    """
    Runs solving routine and compares result to exact solution.
    """
    n = 10
    h = length/(n+1)
    ys = [0 for _ in range(n)]
    fs = [f() for _ in range(n)]

    bs = list(map(lambda x: x * ((h**4) / (E*I)), fs))
    num_iterations = solve(ys, bs, max_iter=1000000, tol=1e-6)
    print("\tNumber of Iterations taken: ", num_iterations)
    exact_plot = compute_exact(np.arange(0, 10.1, 0.1))
    exact_comp = compute_exact(np.linspace(0, length, n+2))
    ys = [0] + ys + [0]
    error, index = compute_error(ys, exact_comp)
    print("\tMax error: %s at index: %s of %s"%(error, index, len(ys)-1))
    plt.figure("Comparing numerical solution with n=10 to exact solution")
    plt.plot(np.linspace(0, length, n+2), [-x for x in ys], color='red')
    plt.plot(np.arange(0, 10.1, 0.1), [- x for x in exact_plot], color='green')
    plt.plot(index*h, -ys[index], 'o', color='blue')
    plt.legend(('numerical solution', 'exact solution', 'position of max error'), loc='best')
    plt.title("\tGauss-Seidel for $n=10$ and $TOL=10^{-6}$, num of iterations = %s" % num_iterations)
    plt.grid()
    #plt.show()


def main_multi():
    """
    Computes condition values for k up to 10, with max_iter = 10000 and tol = 1e-16, and plots results.
    Uses threaded algorithm multi_solve().
    """
    multi_solve(5, max_iter=10000, tol=1e-16)


def main_extra():
    """
    Runs solving routine with extra weight on right side and compares to numerical solution
    without extra weight.
    """
    n = 10
    h = length/(n+1)
    ys = [0 for _ in range(n)]
    ys2 = list(ys)
    fs = [f() for _ in range(n)]
    fs2 = list(fs)
    apply_weight_to_f(fs, h)
    bs = list(map(lambda x: x * ((h**4) / (E*I)), fs))
    bs2 = list(map(lambda x: x * ((h**4) / (E*I)), fs2))
    num_iterations = solve(ys, bs, max_iter=1000000, tol=1e-18)
    num_iterations2 = solve(ys2, bs2, max_iter=1000000, tol=1e-18)
    ys = [0] + ys + [0]
    ys2 = [0] + ys2 + [0]
    max_value = max(ys)
    max_index = ys.index(max_value)
    print("\tIndex of most bended point of bar: %s with %s"%(max_index, max_value))
    plt.figure("Comparing numerical solution with extra weight to standard")
    plt.plot(np.linspace(0, length, n+2), [-x for x in ys], color='green')
    plt.plot(np.linspace(0, length, n+2), [- x for x in ys2], color='red')
    plt.legend(('extra weight', 'standard'), loc='best')
    plt.title("Gauss-Seidel for $n=10$ and $TOL=10^{-18}$, num of iterations = %s" % num_iterations)
    plt.grid()
    #plt.show()


def main_extra_multi():
    """
    Computes condition values for k up to 10, with max_iter = 10000 and tol = 1e-16, and plots results.
    Uses threaded algorithm multi_solve2().
    """
    multi_solve2(5, max_iter=10000, tol=1e-16)


def main_cond():
    """
    Computes condition values for k up to 3 and plots results
    """
    k_limit = 3
    conds = compute_conditions(k_limit)
    plt.figure('conditions Plot')
    plt.title('Condition values for k = 1 to %s' % k_limit)
    plt.plot(range(1, k_limit+1, 1), conds)
    plt.grid()
    #plt.show()


def start():
    print('ANALYZING RESULTS FOR N=10')
    main()
    print('ANALYZING RESULTS FOR k FROM 1 TO 5')
    main_multi()
    print('ANALYZING RESULTS FOR N=10 WITH EXTRA WEIGHT')
    main_extra()
    print('ANALYZING RESULTS FOR k FROM 1 TO 5 WITH EXTRA WEIGHT')
    main_extra_multi()
    main_cond()
    print('DONE')
    plt.show()





start()
