# раскладывает функцию в ряд Фурье,
# и анимацию делает :)
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def f(x):
    return x ** 3


# Параметры разложения
L = 10  # Полупериод (интервал [-L, L])
max_terms = 20  # Максимальное количество гармоник

x = np.linspace(-L, L, 1000)


def coeffsFurie():
    """Вычисляет все коэффициенты Фурье для функции f(x).
       :return:
       - a0: постоянная составляющая (float)
       - a_coeffs: список коэффициентов a_n для косинусных гармоник
       - b_coeffs: список коэффициентов b_n для синусных гармоник"""

    # Вычисляем a₀ (постоянную составляющую)
    a0 = (1 / (2 * L)) * np.trapezoid(f(x), x)  # Среднее значение функции

    # Вычисляем коэффициенты aₙ и bₙ
    a_coeffs = []
    b_coeffs = []
    for n in range(1, max_terms + 1):
        a_n = (1 / L) * np.trapezoid(f(x) * np.cos(n * np.pi * x / L), x)
        b_n = (1 / L) * np.trapezoid(f(x) * np.sin(n * np.pi * x / L), x)
        a_coeffs.append(a_n)
        b_coeffs.append(b_n)

    return a0, a_coeffs, b_coeffs


# Предварительный расчет коэффициентов
a0, a_coeffs, b_coeffs = coeffsFurie()
print(a_coeffs)
print(b_coeffs)

def fourier_series(x, n_terms):
    """Вычисляет частичную сумму ряда Фурье.
       На вход:
       - x: массив точек для вычисления
       - n_terms: количество учитываемых гармоник
       На выход: массив значений частичной суммы ряда"""

    result = a0 * np.ones_like(x) # массив длины x, заполненный a0
    # Добавляем гармоники
    for n in range(1, n_terms + 1):
        result += a_coeffs[n - 1] * np.cos(n * np.pi * x / L)
        result += b_coeffs[n - 1] * np.sin(n * np.pi * x / L)

    return result


# Настройка графики
fig, ax = plt.subplots(figsize=(12, 6))


# Первоначальный график ряда Фурье (1 гармоника)
line, = ax.plot(x, fourier_series(x, 1), 'r--', label='Ряд Фурье')
# График исходной функции
ax.plot(x, f(x), 'b-', label='Исходная функция')
ax.set_title(f'Разложение {f.__name__}(x) в ряд Фурье')
ax.legend()
ax.grid(True)


def update(n):
    """Функция обновления анимации.
    n: количество гармоник
    return: обновленная линия графика"""
    line.set_ydata(fourier_series(x, n))  # Обновляем данные
    ax.set_title(f'{f.__name__}(x): ряд Фурье (n={n})')
    return line,


# Создание и запуск анимации
animation = FuncAnimation(
    fig, # фигура для анимации
    update, # функция, которая обновляет анимацию
    frames=range(1, max_terms + 1),  # Кадры от 1 до max_terms и автоматическая передача в функцию update
    interval=500,  # Задержка между кадрами (мс)
    blit=False
)
plt.show()
