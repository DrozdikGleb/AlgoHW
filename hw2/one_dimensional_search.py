import math


def exhaustive(func, left, right, eps=1e-3):
    iterations = int((right - left) / eps + 10)
    delta = (right - left) / iterations
    min_x = left
    min_f = func(left)
    for k in range(iterations):
        x = left + delta * k
        f_x = func(x)
        if min_f > f_x:
            min_f = f_x
            min_x = x
    return min_x, iterations


def dichotomy(func, left, right, eps=1e-3):
    delta = eps / 3
    iterations = 0
    while right - left > eps:
        iterations += 1
        mid = (right + left) / 2
        x1 = mid - delta
        x2 = mid + delta

        f1 = func(x1)
        f2 = func(x2)

        if f1 < f2:
            right = x2
        elif f1 > f2:
            left = x1
        else:
            left = x1
            right = x2
    return (right + left) / 2, iterations


def golden_ratio(func, left, right, eps=1e-5):
    iterations = 1
    phi = (math.sqrt(5) + 1) / 2

    x1 = left + (2 - phi) * (right - left)
    x2 = right - (2 - phi) * (right - left)

    f1 = func(x1)
    f2 = func(x2)
    while right - left > eps:
        iterations += 1
        if f1 < f2:
            right = x2
            x2 = x1
            # не нужно заново считать f2
            f2 = f1
            x1 = left + (2 - phi) * (right - left)
            f1 = func(x1)
        elif f1 > f2:
            left = x1
            x1 = x2
            f1 = f2
            x2 = right - (2 - phi) * (right - left)
            f2 = func(x2)
        else:
            left = x1
            right = x2
    return (left + right) / 2, iterations
