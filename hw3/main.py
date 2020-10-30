import numpy as np
from scipy import optimize


def linear_func(x, a, b):
    return a * x + b


def linear_grad_a_func(x, a, b):
    return x


def linear_grad_b_func(x, a, b):
    return 1


def rational_func(x, a, b):
    return a / (1 + b * x)


def rational_grad_a_func(x, a, b):
    return 1 / (1 + b * x)


def rational_grad_b_func(x, a, b):
    return -a * x / ((1 + b * x) ** 2)


def rational_grad_b_a_func(x, a, b):
    return -x / ((1 + b * x) ** 2)


def rational_grad_b_b_func(x, a, b):
    return 2 * a * x * x / ((1 + b * x) ** 3)


def rational_grad_a_a_func(x, a, b):
    return 0


def rational_grad_a_b_func(x, a, b):
    return -x / (1 + b * x)


def d(data, a, b, approx_func):
    return sum([(approx_func(x, a, b) - y) ** 2 for x, y in data])


def d_grad(data, a, b, approx_func, approx_first_func):
    return sum([2 * (approx_func(x, a, b) - y) * approx_first_func(x, a, b) for x, y in data])

# F = (f(x) - y) ^ 2
# F`a = 2 * (f(x) - y)
# F`b = 2 * (f(x) - y)
def d_grad_a_b(data, approx_func, approx_a_func, approx_b_func):
    grad_a = lambda a, b: d_grad(data, a, b, approx_func, approx_a_func)
    grad_b = lambda a, b: d_grad(data, a, b, approx_func, approx_b_func)
    f = lambda a, b: d(data, a, b, approx_func)
    return f, grad_a, grad_b


def d_grad_second(data, f, f_grad_a, f_grad_a_a, f_grad_a_b, f_grad_b, f_grad_b_a, f_grad_b_b):
    f1 = lambda a, b: sum([(f(x_p, a, b) - y_p) ** 2 for (x_p, y_p) in data])
    f1_grad_a = lambda a, b: sum(
        [(f(x_p, a, b) - y_p) * 2 * f_grad_a(x_p, a, b) for (x_p, y_p) in data])
    f1_grad_a_a = lambda a, b: sum(
        [(f(x_p, a, b) - y_p) * 2 * f_grad_a_a(x_p, a, b) + 2 * (f_grad_a(x_p, a, b) ** 2)
         for (x_p, y_p) in data])
    f1_grad_a_b = lambda a, b: sum(
        [(f(x_p, a, b) - y_p) * 2 * f_grad_a_b(x_p, a, b) + 2 * f_grad_a(x_p, a, b) * f_grad_b(x_p, a, b)
         for (x_p, y_p) in data])

    f1_grad_b = lambda a, b: sum(
        [(f(x_p, a, b) - y_p) * 2 * f_grad_b(x_p, a, b) for (x_p, y_p) in data])
    f1_grad_b_b = lambda a, b: sum(
        [(f(x_p, a, b) - y_p) * 2 * f_grad_b_b(x_p, a, b) + 2 * (f_grad_b(x_p, a, b) ** 2)
         for (x_p, y_p) in data])
    f1_grad_b_a = lambda a, b: sum(
        [(f(x_p, a, b) - y_p) * 2 * f_grad_b_a(x_p, a, b) + 2 * f_grad_a(x_p, a, b) * f_grad_b(x_p, a, b)
         for (x_p, y_p) in data])

    return f1, f1_grad_a, f1_grad_b, f1_grad_a_a, f1_grad_a_b, f1_grad_b_a, f1_grad_b_b


def minimize_lambda(a, b, f_a_b, f_grad_a, f_grad_b):
    # argmin l: F(x + l * grad F(x))
    # argmin l: F(a + l * grad_a(x), b + l * grad_b(x))

    min_func = lambda l: f_a_b(a - l * f_grad_a(a, b), b - l * f_grad_b(a, b))
    # _, _, _, argmim, _ = dm.golden_search(lambda l: f_a_b(a - l * f_grad_a(a, b), b - l * f_grad_b(a, b)), -1, 1)
    # return argmim
    return optimize.golden(min_func, brack=(-1, 1))


def fast_gradient_descent(func, a, b, func_grad_a, func_grad_b, eps=1e-3):
    step_num = 0
    #y = func(x)
    points = [(a, b)]
    step = 1e-5
    while True:
        step_num += 1
        cur_a, cur_b = points[-1]
        y = func(cur_a, cur_b)
        step = minimize_lambda(cur_a, cur_b, func, func_grad_a, func_grad_b)
        next_a = cur_a - step * func_grad_a(cur_a, cur_b)
        next_b = cur_b - step * func_grad_b(cur_a, cur_b)
        next_y = func(next_a, next_b)
        if abs(next_y - y) < eps:
            return step_num, next_y, points
        points.append((next_a, next_b))


def conjugate_gradient_descent(func, a, b, func_grad_a, func_grad_b):
    fprime = lambda v: np.asarray((func_grad_a(v[0], v[1]), func_grad_b(v[0], v[1])))
    f = lambda v: func(v[0], v[1])
    x0 = np.asarray((a, b))
    res = optimize.fmin_cg(f, x0=x0, fprime=fprime)
    return res[0], res[1]


def newton_method(data, f, f_grad_a, f_grad_b, f_grad_a_a, f_grad_a_b, f_grad_b_a, f_grad_b_b):
    # def d_grad_second(data, f, f_grad_a, f_grad_a_a, f_grad_a_b, f_grad_b, f_grad_b_a, f_grad_b_b):
    func, func_grad_a, func_grad_b, func_grad_a_a, func_grad_a_b, func_grad_b_a, func_grad_b_b =  \
        d_grad_second(data, f, f_grad_a, f_grad_a_a, f_grad_a_b, f_grad_b, f_grad_b_a, f_grad_b_b)
    fprime = lambda v: np.asarray((func_grad_a(v[0], v[1]), func_grad_b(v[0], v[1])))

    fprime2 = lambda v: np.asarray(((func_grad_a_a(v[0], v[1]), func_grad_a_b(v[0], v[1])),
                                    (func_grad_b_a(v[0], v[1]), func_grad_b_b(v[0], v[1]))))
    f = lambda v: func(v[0], v[1])
    x0 = np.asarray((0.1, 0.2))
    res = optimize.minimize(f, x0=x0, jac=fprime, hess=fprime2,
                            method="Newton-CG")
    print(res)
    return res.x[0], res.x[1]


def levenberg(data, func):
    f1 = lambda ab: [(func(x_p, ab[0], ab[1]) - y_p) ** 2 for (x_p, y_p) in data]

    res = optimize.least_squares(f1, (1, 1), method='lm')
    return res


if __name__ == '__main__':
    pass
