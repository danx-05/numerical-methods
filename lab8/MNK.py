import numpy as np
import matplotlib.pyplot as plt

from Gauss import Gauss
# https://ru.wikipedia.org/wiki/Метод_наименьших_квадратов

def sum_power(x, p):
    """
    Суммирует элементы массива x в степени p
    """
    s = 0
    for i in x:
        s += i ** p
    return s


def getMatrX(x, p):
    """
    Строит матрицу по массиву x, нужную для МНК, размерности p+1 на p+1
    """
    X = np.zeros((p + 1,p + 1))
    for i in range(2 * p + 1): # обход по диагоналям
        if i == 0:
            X[i,i] = len(x)
        else:
            d = sum_power(x,i) # вычисляет сумму i-ых степеней массива x
            for j in range(i+1):
                k = i - j
                if k <= p and j <= p:
                    X[j,k] = d
    return X
def sumXY(x,y,p):
    """
    Суммирует сумму виды x_t^p * y_t
    :return:
    """
    summ = 0
    for i in range(len(x)):
        summ += x[i] ** p * y[i]
    return summ


def getMatrY(x,y,p):
    """
    Строит вектор свободных членов, нужных для вычисления МНК
    :param x:
    :param y:
    :param p:
    :return:
    """
    Y = np.zeros(p+1)
    Y[0] = sum(y)
    for i in range(1, p + 1):
        Y[i] = sumXY(x,y,i)
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
    n = 25
    p = 3
    x_0 = -4
    x_n = 4
    x_start = np.linspace(x_0, x_n, n)
    y_start = np.sin(x_start)

    # создаем матрицы для МНК
    A = getMatrX(x_start, p)
    b = getMatrY(x_start, y_start,p)
    print("B", b)

    # решаем систему методом Гаусса
    ans = Gauss(A,b)
    print(ans)


    # создаем апроксимированные новый данные многочленом степень p
    x_new = np.linspace(x_0, x_n, 1000)
    y_new = np.zeros_like(x_new)
    for i in range(len(y_new)):
        y_new[i] = value(ans, x_new[i])
    y_sin = np.sin(x_new)
    error = y_new - y_sin
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(x_new, y_new)
    ax1.plot(x_new, y_sin)
    ax1.scatter(x_start,y_start, color = "red")
    ax1.grid()

    ax2.plot(x_new, error)
    ax2.grid()

    plt.tight_layout()  # Чтобы графики не перекрывались
    plt.show()
