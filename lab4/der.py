import math
import numpy as np
from matplotlib import pyplot as plt
def f(x):
    """
    Вычисление значения в точке x
    """
    return math.sin(x)
def right(x, h):
    """
    Вычисление правой производной в точке x
    """
    return (f(x+h) - f(x)) / h
def left(x, h):
    """
    Вычисление левой производной в точке x
    """
    return - (f(x-h) - f(x)) / h
def middle(x,h):
    """
    Вычисление средней производной в точке x
    """
    return (f(x+h) - f(x-h)) / (2 * h)

def right_2(x, h):
    """
    Вычисление второй производной в точке x
    """
    return (middle(x+h,h) - middle(x,h)) / h


h = 10 ** (-6)
x = np.linspace(0, 2 * math.pi, 100)
y1 = np.ones(100)
y2 = np.ones(100)
y3 = np.ones(100)

y4 = np.cos(x)
y5 = - np.sin(x)
for i in range(len(x)):
    y1[i] = right(x[i],h)
    y2[i] = middle(x[i], h)
    y3[i] = right_2(x[i],h)
y6 = y1 - y4
plt.plot(x,y1,"red", label = "правая производная")
plt.plot(x,y2, "green", label = "центральная производная")
plt.plot(x,y3, "blue", label = "вторая производная")
plt.plot(x,y4,"yellow", label = "cos(x)")
plt.plot(x,y5,"black", label = "sin(x)")
#plt.plot(x, y6)
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.legend()
plt.show()


