import math

import numpy as np
from matplotlib import pyplot as plt


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
if __name__ == "__main__":
    start = -5
    end = 5
    n = 30
    # стартовые точки
    #x_start = np.linspace(start, end, n) # равномерная сетка
    # Чебышевская сетка
    x_start = np.ones(n)
    for i in range(n):
        # формула нахождения узлов Чебышева
        x_start[i] = 1/2 * (start + end) + 1/2 * (end - start) * math.cos((2 * (i+1) - 1) / (2*n) * math.pi )
    y_start = np.abs(np.sin(x_start))

    x_new = np.linspace(start, end, 500)
    y_new = np.zeros(len(x_new))
    # коэффициенты многочлена Ньютона
    coeffs = newton_interpolation(x_start, y_start)

    # вычисление по многочлену Ньютона новых точек
    for i in range(len(x_new)):
        y_new[i] = valueInX(coeffs, x_new[i], x_start)
    # построение графиков
    plt.figure(figsize=(10, 6))
    plt.plot(x_new, y_new, linewidth = 3, color = "orange", label="Интерполяция Ньютона", linestyle="--")

    x_sin = np.linspace(start, end, 500)
    y_sin = np.abs(np.sin(x_sin))
    plt.scatter(x_start, y_start, color="red", label="Узлы интерполяции")
    plt.plot(x_sin, y_sin)
    plt.plot(x_sin, y_sin, color = "blue", label="Исходная функция $|\sin(x)|$", linewidth=2)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Интерполяция $|\sin(x)|$ методом Ньютона")
    plt.legend()
    plt.grid(True)
    plt.show()
