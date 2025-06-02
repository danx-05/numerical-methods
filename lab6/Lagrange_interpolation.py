import math

import numpy as np
import matplotlib.pyplot as plt


def lagrange_interpolation(x_start, y_start, x):
    """
    Подсчитывает интерполированное значение x
    :param x_start: начальные данные по x
    :param y_start: начальные данные по y
    :param x: точка, в которой нужно посчитать
    :return: значение в точке x
    """
    n = len(x_start)
    result = 0.0

    for i in range(n):
        # берем значение функции - y
        term = y_start[i]
        # и домножаем его на многочлен Лагранжа
        for j in range(n):
            if j != i:
                term *= (x - x_start[j]) / (x_start[i] - x_start[j])
        # суммируем полученные значения
        result += term
    return result
if __name__ == "__main__":
    n = 43
    start = -5
    end = 5
    # x_start = np.linspace(0, 10, n)
    x_start = np.ones(n)
    for i in range(n):
        x_start[i] = 1 / 2 * (start + end) + 1 / 2 * (end - start) * math.cos((2 * (i + 1) - 1) / (2 * n) * math.pi)
    y_start = np.abs(np.sin(x_start))
    # Точки для построения графика
    x_plot = np.linspace(start, end, 500)
    y_plot = np.abs(np.sin(x_plot))
    y_interp = [lagrange_interpolation(x_start, y_start, x) for x in x_plot]

    # Визуализация
    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, y_plot, color = "blue", label="Исходная функция $|\sin(x)|$", linewidth=2)
    plt.plot(x_plot, y_interp, linewidth = 3, color = "orange", label="Интерполяция Лагранжа", linestyle="--")
    plt.scatter(x_start, y_start, color="red", label="Узлы интерполяции")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Интерполяция $|\sin(x)|$ методом Лагранжа")
    plt.legend()
    plt.grid(True)
    plt.show()
