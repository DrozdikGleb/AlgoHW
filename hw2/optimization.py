import numpy as np
from scipy import optimize

EPS = 0.001


def exhaustive_search(data, x_0, x_1, y_0, y_1, approximation):
    iter_x = 110
    iter_y = 110
    delta_x = (x_1 - x_0) / iter_x
    delta_y = (y_1 - y_0) / iter_y

    best_pair = (2 ** 100, x_0, y_0)

    for a in range(iter_x):
        for b in range(iter_y):
            least_square_error = 0
            a_coord = x_0 + a * delta_x
            b_coord = y_0 + b * delta_y
            for x_pos, y_pos in data:
                least_square_error += (approximation(x_pos, a_coord, b_coord) - y_pos) ** 2

            if least_square_error < best_pair[0]:
                best_pair = (least_square_error, a_coord, b_coord)
    return best_pair, iter_x


def gauss_method(iter_count, a, b, f):
    current_iter = 0
    points = [(a, b)]
    ls = [-1]
    while True:
        cur_a, cur_b = points[-1]
        if len(points) % 2 == 0:
            l = optimize.golden(lambda l1: f(l1, cur_b), brack=(-1, 1))
            next_a = l
            next_b = cur_b
        else:
            l = optimize.golden(lambda l1: f(cur_a, l1), brack=(-1, 1))
            next_a = cur_a
            next_b = l
        ls.append(l)
        points.append((next_a, next_b))
        current_iter += 1
        if abs(f(cur_a, cur_b) - f(next_a, next_b)) < EPS or current_iter > iter_count:
            break
    return points, ls, current_iter


def nelder_mead(func_a_b):
    res = optimize.minimize(lambda x: func_a_b(x[0], x[1]), np.array((0, 0)),
                            method='Nelder-Mead')
    print(res)

    return [], res.x, 0
