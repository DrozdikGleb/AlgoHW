import numpy as np
from scipy import optimize


def rational_func(x, a, b, c, d):
    return (a * x + b) / (x ** 2 + c * x + d)


def d_func(data):
    return lambda a, b, c, d: sum([(rational_func(x, a, b, c, d) - y) ** 2 for (x, y) in data])


def generate_data():
    f_x = lambda x: 1 / (x ** 2 - 3 * x + 2)
    x_s = [3 * i / 1000 for i in range(1000)]
    t_s = np.random.normal(size=1000)
    y_s = []
    for (x, t) in zip(x_s, t_s):
        f_k = f_x(x)
        if f_k < -100:
            y_s.append(-100 + t)
        elif f_k <= 100:
            y_s.append(f_k + t)
        else:
            y_s.append(100 + t)
    return list(zip(x_s, y_s))


def nelder_mead(data):
    d = d_func(data)
    res = optimize.minimize(lambda x: d(x[0], x[1], x[2], x[3]), np.array((0.1, 0.2, 0.3, 0.4)),
                            method='Nelder-Mead')
    print(res)
    return res.x


def levenberg_marquardt(data):
    d_func = lambda el: [(rational_func(x, el[0], el[1], el[2], el[3]) - y) ** 2 for (x, y) in data]

    res = optimize.least_squares(d_func, (0.1, 0.2, 0.3, 0.4), method='lm')
    print(res)
    return res.x


def differential_evolution(data):
    d = d_func(data)
    res = optimize.differential_evolution(lambda x: d(x[0], x[1], x[2], x[3]),
                                          ((-2, 2), (-2, 2), (-2, 2), (-2, 2)))
    print(res)
    return res.x


def simultaneous_anneal(data):
    d = d_func(data)
    res = optimize.dual_annealing(lambda x: d(x[0], x[1], x[2], x[3]),
                                  ((-2, 2), (-2, 2), (-2, 2), (-2, 2)), maxiter=1000)
    print(res)
    return res.x

