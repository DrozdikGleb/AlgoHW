from functools import reduce
from random import randint
from typing import List


def const_func(data: List):
    return 1


def sum_func(data: List):
    return sum(data)


def product_func(data: List):
    return reduce(lambda a, b: a * b, data)


def polynom_func(data: List):
    pol_sum = 0
    for i in range(len(data)):
        pol_sum += data[i] * pow(1.5, i)
    return pol_sum


def horner_polynom_func(data: List):
    pol_sum = 0
    prev_sum = data[len(data) - 1]
    for i in range(len(data) - 1, 0, -1):
        prev_elem = data[i - 1]
        pol_sum = prev_elem + 1.5 * prev_sum
        prev_sum = pol_sum
    return pol_sum


def bubble_sort(data: List):
    for i in range(len(data)):
        for j in range(i, len(data)):
            if data[i] > data[j]:
                data[i], data[j] = data[j], data[i]


def quick_sort_inner(data: List, fst, lst):
    if fst >= lst:
        return

    i, j = fst, lst
    m = randint(fst, lst)
    while i <= j:
        while data[i] < data[m]:
            i += 1
        while data[j] > data[m]:
            j -= 1
        if i <= j:
            data[i], data[j] = data[j], data[i]
            i, j = i + 1, j - 1
    quick_sort_inner(data, fst, j)
    quick_sort_inner(data, i, lst)


def quick_sort(data: List):
    quick_sort_inner(data, 0, len(data) - 1)


def tim_sort(data: List):
    data.sort()


def matrix_gen(n):
    matr = []
    for i in range(n):
        matr.append([randint(1, 100) for _ in range(n)])
    return matr

def matrix_multiplication(data: List):
    res = [[0 for _ in range(len(data))] for _ in range(len(data))]

    for i in range(len(data)):
        for j in range(len(data[0])):
            for k in range(len(data)):
                res[i][j] += data[i][k] * data[k][j]


if __name__ == '__main__':
    print(polynom_func([10, 7, 14, 2, 4]))
    print(horner_polynom_func([10, 7, 14, 2, 4]))
