# Реализованы три метода численного интегрирования для функции sin(x):
# Метод прямоугольников, трапеций, Симпсона
# строятся графики
import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate


def f(x):
    return np.sin(x)


def rectangle(a, b, h):
    """
    Интегрирование методом прямоугольников
    :param b: Верхний предел
    :param h: Точность
    :return: Значение интеграла от 0 до b, массив x c интервалом h, массив y
    """
    total_sum = 0
    x = []
    y = []
    current = a
    while current < b:
        x.append(current)
        S_rectangle = f(current) * h
        total_sum += S_rectangle
        y.append(total_sum)
        current += h
    return total_sum, x, y


def trapeze(a, b, h):
    """
    Интегрирование методом трапеций
    :param b: Верхний предел
    :param h: Точность
    :return: Значение интеграла от 0 до b, массив x c интервалом h, массив y
    """
    total_sum = 0
    x = []
    y = []
    current = a
    while current < b:
        x.append(current+h)
        area = (f(current) + f(current + h)) / 2 * h
        total_sum += area
        y.append(total_sum)
        current += h
    return total_sum, x, y


def simpson(a, b, h):
    """
    Интегрирование методом Симпсона
    :param b: Верхний предел
    :param h: Точность
    :return: Значение интеграла от 0 до b, массив x c интервалом h, массив y
    """
    x = []
    y = []
    n = int((b - a) / h)
    total_sum = 0
    for i in range(0, n, 2):
        area = f(a + i * h) + 4 * f(a+h + i * h) + f(a + 2*h + i * h) # вычисление площади под параболой
        total_sum += area
        x.append(a + i * h + 2 * h)
        y.append(h / 3 * total_sum)
    return h / 3 * total_sum, x, y
if __name__ == "__main__":
    # Параметры интегрирования
    a = 0
    b = 7
    c = 100  # Количество разбиений
    h = (b - a) / c

    # Точное решение для сравнения
    x_real = np.linspace(a, b, 10000)
    y_real = 1 - np.cos(x_real)

    # Вычисление интегралов разными методами
    value_int1, x1, y1 = rectangle(a, b, h)
    value_int2, x2, y2 = trapeze(a, b, h)
    value_int3, x3, y3 = simpson(a, b, h)

    # Точное значение с помощью scipy
    result, error = integrate.quad(np.sin, a, b)

    # Вывод результатов
    print("1) Метод прямоугольников:", value_int1)
    print("2) Метод трапеций:", value_int2)
    print("3) Метод Симпсона:", value_int3)
    print("4) Точное значение:", result)

    # Построение графиков
    plt.figure(figsize=(8, 5))
    plt.plot(x1, y1, "red", label="Метод прямоугольников")
    plt.plot(x2, y2, "green", label="Метод трапеций")
    plt.plot(x3, y3, "blue", label="Метод Симпсона")
    plt.plot(x_real, y_real, "black", label="Точное решение: 1 - cos(x)")

    plt.title("Сравнение методов численного интегрирования")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()

