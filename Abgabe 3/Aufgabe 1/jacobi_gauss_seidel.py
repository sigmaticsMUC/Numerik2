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
    return [(f()*(x**2)*(length - x)**2)/(24 * E * I) for x in space]


def f():
    return density*width*depth*g


def apply_weight_to_f(fs, h):
    for i in range(len(fs)):
        if 3 <= h*i <= 4:
            fs[i] += 500
            

def create_matrix(n):
    matrix = [[] for _ in range(n)]
    matrix[0] = [12, -6, 4.0/3.0] + [0 for _ in range(n-3)]
    matrix[1] = [-4, 6, -4, 1] + [0 for _ in range(n-4)]
    matrix[n-2] = [0 for _ in range(n-4)] + [1, -4, 6, -4]
    matrix[n-1] = [0 for _ in range(n-3)] + [4.0/3.0, -6, 12]
    for j in range(2, n-2, 1):
        matrix[j] = [0 for _ in range(j-2)] + [1, -4,  6, -4, 1] + [0 for _ in range(n-3-j)]
    return matrix


def inner_sum(row, ys, index):
    result = 0
    for i in range(len(ys)):
        if i != index:
            result += row[i]*ys[i]
    return result


def inner_sum2(ys, index):
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


def seidel2(ys, bs):
    ys_old = list(ys)
    for i in range(len(ys)):
        diag_elem = 6.0
        if i == 0 or i == len(ys)-1:
            diag_elem = 12.0
        ys[i] = (bs[i] - inner_sum2(ys, i))/diag_elem
    return ys, ys_old


def seidel(ys, matrix, bs):
    ys_old = list(ys)
    for i in range(len(ys)):
        ys[i] = (bs[i] - inner_sum(matrix[i], ys, i))/matrix[i][i]
    return ys, ys_old


def solve(ys, bs, max_iter=40000, tol=1e-6):
    it_counter = 0
    iterating = True
    while iterating:
        _, ys_old = seidel2(ys, bs)
        step = max([a_i - b_i for a_i, b_i in zip(ys, ys_old)])
        it_counter += 1
        if it_counter >= max_iter or abs(step) <= tol:
            iterating = False
    return it_counter


def compute_error(numerical, exact):
    difference = [abs(a_i - b_i) for (a_i, b_i) in zip(exact, numerical)]
    max_error = max(difference)
    max_error_index = difference.index(max_error)
    return max_error, max_error_index


def compute_conditions(k_limit):
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
        plt.plot(line_space_list[index], [-x for x in ys_list[index]])
        error , error_index = compute_error(ys_list[index], compute_exact(line_space_list[index]))
        print('Max error: %s for k=%s: at index: %s of %s'%(error, index+1, error_index, len(ys_list[index])-1))
    end = time.time()
    plt.title('Plot for k = 1 to 10 with TOL=%s'%tol)
    plt.legend(('k=1', 'k=2','k=3','k=4','k=5','k=6','k=7','k=8','k=9','k=10'), loc='best')
    print("time needed in seconds: ", end - start)
    plt.show()


def multi_solve2(k_limit, max_iter=10000, tol=1e-10):
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
        plt.plot(line_space_list[index], [-x for x in ys_list[index]])
        max_bend = max(ys_list[index])
        max_bend_pos = list(ys_list[index]).index(max_bend)
        plt.plot(max_bend_pos * (length/(len(ys_list[index])+1)), -max_bend, 'o')
    end = time.time()
    plt.title('Plot for k = 1 to %s with TOL=%s'%(k_limit,tol))
    print("time needed in seconds: ", end - start)
    plt.show()


def main():
    n = 10
    h = length/(n+1)
    ys = [0 for _ in range(n)]
    fs = [f() for _ in range(n)]

    bs = list(map(lambda x: x * ((h**4) / (E*I)), fs))
    num_iterations = solve(ys, bs, max_iter=1000000, tol=1e-6)
    print('Number of Iterations taken: ', num_iterations)
    exact_plot = compute_exact(np.arange(0, 10.1, 0.1))
    exact_comp = compute_exact(np.linspace(0, length, n+2))
    ys = [0] + ys + [0]
    error, index = compute_error(ys, exact_comp)
    print('Max error: %s at index: %s of %s'%(error, index, len(ys)-1))
    plt.figure("Comparing numerical solution with n=10 to exact solution")
    plt.plot(np.linspace(0, length, n+2), [-x for x in ys], color='red')
    plt.plot(np.arange(0, 10.1, 0.1), [- x for x in exact_plot], color='green')
    plt.plot(index*h, -ys[index], 'o', color='blue')
    plt.legend(('numerical solution', 'exact solution', 'position of max error'), loc='best')
    plt.title("Gauss-Seidel for $n=10$ and $TOL=10^{-6}$, num of iterations = %s"%(num_iterations))
    plt.grid()
    plt.show()


def main2():
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
    print('Index of most bended point of bar: %s with %s'%(max_index, max_value))
    plt.figure("Comparing numerical solution with extra weight to standard")
    plt.plot(np.linspace(0, length, n+2), [-x for x in ys], color='green')
    plt.plot(np.linspace(0, length, n+2), [- x for x in ys2], color='red')
    plt.legend(('extra weight', 'standard'), loc='best')
    plt.title("Gauss-Seidel for $n=10$ and $TOL=10^{-18}$, num of iterations = %s"%num_iterations)
    plt.grid()
    plt.show()


def main_cond():
    k_limit = 3
    conds = compute_conditions(k_limit)
    plt.plot(range(1, k_limit+1, 1), conds)
    plt.title('Condition values for k = 1 to %s'%k_limit)
    plt.show()

#main_cond()
#main()
#main2()
multi_solve2(5, max_iter=50000, tol=1e-18)
#print(compute_conditions(1))