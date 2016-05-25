from scipy.constants import g
from matplotlib import pyplot as plt
import numpy as np
import threading as thr
import time



length = 10.0
width = 0.1
depth = 0.05
density = 7850
E = 2e11
I = (width*(depth**3))/12.0


def f():
    return density*width*depth*g


#def fs_with_weight(weight, h):
#  berechne wie viele nodes betroffen sind und welche kraftlast auf jede einzelne node wirkt
#  addiere diese zu bs dazu


def create_band_matrix(n):
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


def seidel(ys, matrix, bs):
    ys_old = list(ys)
    for i in range(len(ys)):
        ys[i] = (bs[i] - inner_sum(matrix[i], ys, i))/matrix[i][i]
    return ys, ys_old


def solve(ys, bs, max_iter=40000, tol=1e-20):
    it_counter = 0
    iterating = True
    coefficient_matrix = create_band_matrix(len(ys))
    while iterating:
        _, ys_old = seidel(ys, coefficient_matrix, bs)
        step = max([a_i - b_i for a_i, b_i in zip(ys, ys_old)])
        it_counter += 1
        if it_counter >= max_iter or step <= tol:
            iterating = False
    return it_counter


def compute_error(numerical, exact):
    difference = [abs(a_i - b_i) for a_i, b_i in zip(exact, numerical)]
    max_error = max(difference)
    max_error_index = difference.index(max_error)
    return max_error, max_error_index


def multi_solve(k_limit, max_iter=10000, tol=1e-16, max_thread_count = 100):

    start = time.time()
    n_list = [10 * (2**k) + 1 for k in range(1, k_limit, 1)]
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
        thread_list[index] = thr.Thread(target=solve, args=(ys_list[index], bs, ))
        thread_list[index].start()

    for thread in thread_list:
        index = thread_list.index(thread)
        thread.join()
        #plt.plot(line_space_list[index], [0] + [-x for x in ys_list[index]] + [0])
    end = time.time()
    print(end - start)
    #plt.show()


def main():
    n = 81
    h = length/(n+1)
    ys = [0 for _ in range(n)]
    fs = [f() for _ in range(n)]

    bs = list(map(lambda x: x * ((h**4) / (E*I)), fs))
    solve(ys, bs)
    plt.plot(np.linspace(0, length, n+2), [0] + [-x for x in ys] + [0])
    #plt.plot(np.arange(0, 10.1, 0.1), [-(f()*(x**2)*(length - x)**2)/(24 * E * I) for x in np.arange(0, 10.1, 0.1)])
    plt.grid()
    plt.show()


#main()
multi_solve(5)
307.4121398925781