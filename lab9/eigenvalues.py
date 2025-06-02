# тот, который на лекции был
# поиск собственных значений матрицы

from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt


def determinant(x):
    """
    Вычисление определителя матрицы с использованием разложения по строке
    (поддерживает массивы NumPy)
    """
    n = x.shape[0]
    if n == 2:
        return x[0, 0] * x[1, 1] - x[0, 1] * x[1, 0]

    summ = 0
    for i in range(n):
        # Создаем подматрицу без первой строки и i-го столбца
        submatrix = np.delete(np.delete(x, 0, axis=0), i, axis=1)
        summ += x[0, i] * (-1) ** i * determinant(submatrix)
    return summ
def valueInX(coeffs, x, x_nodes):
    """
    Вычисляет значение многочлена Ньютона в заданной точке x
    :param coeffs: коэффициенты многочлена Pn(x) = a_0 + a_1 * (x - x_0) + ... + a_n * (x - x_0)(x - x_1) ... (x - x_{n-1})
    :param x:
    :param x_nodes: начальные точки х-ов
    :return: значение многочлена в точке
    """
    y = 0
    for i in range(len(coeffs)):
        p = coeffs[i]
        for j in range(0,i):
            p *= (x - x_nodes[j])
        y += p
    return y
def solve(a,b,e, coeffs, x):
    """

    :param a: начальное значение отрезка для поиска корня
    :param b: конечное значение отрезка для поиска корня
    :param e: точность приближения
    :param coeffs: коэффициенты
    :return: c - найденный корень, k - количество итераций
    """
    c = (a+b)/2
    k = 0
    while abs(a - b) > e: # Условие остановки алгоритма
        c = (a+b) / 2 # вычисляем середину отрезка   valueInX(coeffs, x_mass[i], x)
        if valueInX(coeffs, a, x) * valueInX(coeffs, c, x) > 0:
            a = c # сдвигаем левый конец до середины
        else:
            b = c # сдвигаем правый конец до середины
        k+=1
    return c
def newton_interpolation(x, y):
    """
    Вычисление коэффициентов многочлена Ньютона
    :param x: начальные данные: точки х
    :param y: начальные данные: точки y
    :return: массив коэффициентов многочлена
    """
    n = len(x)
    coeffs = np.zeros((n, n)) # для вычисления создается матрица, итоговый массив будет лежать в первой ее строке - coeffs[0]
    # по диагонали начальные данные точек y
    # вычисление разделенных разностей
    for i in range(n):
        coeffs[i][i] = y[i]
    # вычисление остальных коэффициентов по реккурентной формуле
    for i in range(n - 2, -1, -1):
        for j in range(i+1, n):
            coeffs[i][j] = (coeffs[i+1][j] - coeffs[i][j-1])/(x[j] - x[i])
    return coeffs[0]


def matrix_vector_multiply(matrix, vector):
    result = [0.0 for _ in range(len(matrix))]

    for i in range(len(matrix)):
        for j in range(len(vector)):
            result[i] += matrix[i][j] * vector[j]

    return result
def scar_mult(u, v):
    """
    Перемножает скалярно <u,v>
    :param u:
    :param v:
    :return:
    """
    summ = 0
    for i in range(len(v)):
        summ += u[i] * v[i]
    return summ

def estimate(v0, v1):
    """

    :param v0:
    :param v1:
    :return:
    """
    # <v1,v1>
    # ------
    # <v1, v0>
    return scar_mult(v1, v1) / scar_mult(v1, v0)
def max_lambda(matr):
    """
    Считает максимальное по модулю собственное значение
    :param matr:
    :return:
    """
    m_lambda = 0
    epsilon = 10**(-6)
    l = len(matr)
    v = [1 for _ in range(l)]
    n = 100000
    # u_{s+1} = A * u_s
    while n:
        # итеративно считает максимальное значение
        temp = matrix_vector_multiply(A, v)
        new_lambda = estimate(v, temp) # подсчитывает новое итеративное значение
        if abs(m_lambda - new_lambda) < epsilon:
            return new_lambda
        m_lambda = new_lambda
        v = temp
        n -= 1

A = np.array([
    [10, 2, 2],
    [2, 4, 5],
    [2, 5, 7]
], dtype=float)
# A = np.array([
#     [5, 1, 0, 0, 0, 0, 0, 0],
#     [1, 5, 1, 0, 0, 0, 0, 0],
#     [0, 1, 5, 1, 0, 0, 0, 0],
#     [0, 0, 1, 5, 1, 0, 0, 0],
#     [0, 0, 0, 1, 5, 1, 0, 0],
#     [0, 0, 0, 0, 1, 5, 1, 0],
#     [0, 0, 0, 0, 0, 1, 5, 1],
#     [0, 0, 0, 0, 0, 0, 1, 5]
# ], dtype=float)

lamd = max_lambda(A)
n = len(A)

# берутся n+1 точка и считается определитель характеристического многочлена матрицы для этих точек
x = np.linspace(-abs(lamd), abs(lamd), n+1)
y = np.zeros(n+1)

ans = []
for i in range(n+1):
    l = x[i]
    B = deepcopy(A)
    for j in range(n):
        B[j][j] -= l
    y[i] = determinant(B)


# производится интерполяция методом Ньютона
coeffs = newton_interpolation(x,y) # коэффициенты многочлена

# строит интерполированные массивы определителя
x_mass = np.linspace(-1-abs(lamd), abs(lamd) + 1, 50)
y_mass = np.zeros_like(x_mass)
for i in range(len(x_mass)):
    y_mass[i] = valueInX(coeffs, x_mass[i], x)

# ищет корни интерполированного многочлена
for i in range(len(x_mass) - 1):
    if y_mass[i] * y_mass[i+1] < 0:
        ans.append(solve(x_mass[i], x_mass[i+1], 10**(-6), coeffs, x)) # метод дихотомии

print(*ans)
eigenvalues = np.linalg.eigvals(A)
print(*np.sort(eigenvalues))


plt.scatter(ans, np.zeros(len(eigenvalues)), color = "red")
plt.plot(x_mass, y_mass, "b")
plt.grid()
plt.show()
