# Подсчет длины дуги эллипса с помощью метода Симпсона, по умолчанию считается для окружности
import math
def f(t, a, b):
    """
    Интеграл для вычисления дуги эллипса
    return: значение функции дуги эллипса в точке t
    """
    return math.sqrt(a**2 * math.sin(t) * math.sin(t) + b**2 * math.cos(t) * math.cos(t))
def simpson(st, end, h, a, b):
    x = []
    y = []
    n = int((end - st) / h)
    total_sum = 0
    for i in range(0, n, 2):
        area = f(st + i * h, a,b) + 4 * f(st + h + i * h, a, b) + f(st + 2 * h + i * h,a,b)
        total_sum += area
        x.append(st + i * h + 2 * h)
        y.append(h / 3 * total_sum)
    return h / 3 * total_sum

print(4 * simpson(0, math.pi / 2, math.pi / 200, 1 , 1))


