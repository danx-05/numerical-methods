import numpy as np
import matplotlib.pyplot as plt

def f(x,y):
    """
    Данная функция
    """
    return -y

def RK2(x_k, y_k, h):
    temp = f(x_k, y_k)
    return y_k + h/2 * (temp + f(x_k + h, y_k + h * temp))


h = 10 ** (-2)           # шаг
x = np.arange(0, 5, h)  # массив x с шагом h
y = np.zeros_like(x)     # массив y
e = np.exp(-x)
y[0] = 1

for i in range(1, len(x)):
    y[i] = RK2(x[i - 1], y[i - 1], h)
error = np.abs(y - e)  # Ошибка в узлах сетки
# Создаем окно с двумя subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# График численного и точного решения
ax1.plot(x, e, color = "red", label='Точное решение ($e^{-x}$)')
ax1.plot(x, y, color = "blue", label=f'Рунге-Кутта 2 (h={h})')
ax1.set_xlabel('x')
ax1.set_ylabel('y(x)')
ax1.legend()
ax1.grid()
ax1.set_title('Численное решение vs Точное решение')

# График ошибки
ax2.plot(x, error, 'g', label='Абсолютная ошибка')
ax2.set_xlabel('x')
ax2.set_ylabel('модуль ошибки')
ax2.legend()
ax2.grid()
ax2.set_title('График ошибки')

plt.tight_layout()  # Чтобы графики не перекрывались
plt.show()
