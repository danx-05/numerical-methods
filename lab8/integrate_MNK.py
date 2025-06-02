import numpy as np
from matplotlib import pyplot as plt
from Gauss import Gauss

# https://dep805.ru/education/portal/4/to/l12_4tochm.pdf
h = 0.0001 # точность для интегрирования

def f(x, p):
    return np.sin(x) * x ** p

def g(x, p):
    return x ** p
def trapeze(a, b, p, h):
    """
    Интегрирование методом трапеций
    :param b: Верхний предел
    :param h: Точность
    :return: Значение интеграла от 0 до b, массив x c интервалом h, массив y
    """
    total_sum = 0
    current = a
    while current < b:
        area = (g(current, p) + g(current + h, p)) / 2 * h
        total_sum += area
        current += h
    return total_sum
def trapeze2(a, b, p, h):
    """
    Интегрирование методом трапеций
    :param b: Верхний предел
    :param h: Точность
    :return: Значение интеграла от 0 до b, массив x c интервалом h, массив y
    """
    total_sum = 0
    current = a
    while current < b:
        area = (f(current, p) + f(current + h, p)) / 2 * h
        total_sum += area
        current += h
    return total_sum

def integral_power(a,b, p):
    return trapeze(a,b,p,h)


def getMatrX(a,b, p):
    """
    Строит матрицу по массиву x, нужную для МНК, размерности p+1 на p+1
    a, b - пределы интегрирования
    """
    X = np.zeros((p + 1,p + 1))
    for i in range(2 * p + 1): # обход по диагоналям
        if i == 0:
            X[i,i] = b - a
        else:
            d = integral_power(a, b, i) # вычисляет сумму i-ых степеней массива x
            for j in range(i+1):
                k = i - j
                if k <= p and j <= p:
                    X[j,k] = d
    return X
def sumXY(x,y,p):
    summ = 0
    for i in range(len(x)):
        summ += x[i] ** p * y[i]
    return summ


def getMatrY(a, b, p):
    """
    Строит вектор свободных членов, нужных для вычисления МНК
    """
    Y = np.zeros(p+1)
    Y[0] = trapeze2(a,b,0,h)
    for i in range(1, p + 1):
        Y[i] = trapeze2(a,b,i,h)
    return Y

def value(coeffs, x):
    """
    Значение многочлена по коэффициентам
    """
    y = 0
    for i in range(len(coeffs)):
        y += coeffs[i] * x ** i
    return y

if __name__ == "__main__":
    # начальные данные


    p = 3
    x_0 = -4
    x_n = 4
    x_sin = np.linspace(x_0,x_n, 1000)
    y_sin = np.sin(x_sin)

    # создаем матрицы для МНК
    A = getMatrX(x_0,x_n, p)
    b = getMatrY(x_0,x_n, p)
    print("B", b)

    # решаем систему методом Гаусса
    ans = Gauss(A,b)
    print(ans)


    # создаем апроксимированные новый данные многочленом степень p
    x_new = np.linspace(x_0, x_n, 1000)
    y_new = np.zeros_like(x_new)
    for i in range(len(y_new)):
        y_new[i] = value(ans, x_new[i])
    error = y_new - y_sin
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(x_new,y_new)
    ax1.plot(x_sin, y_sin)
    ax1.grid()

    ax2.plot(x_sin, error)
    ax2.grid()

    plt.tight_layout()  # Чтобы графики не перекрывались
    plt.show()



