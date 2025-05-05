import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Параметры
g = 9.8        # ускорение свободного падения (м/с²)
y0 = 50        # начальная высота (м)
v0 = 0.0       # начальная скорость (м/с)
t_max = 4.0    # время моделирования (с)
dt = 0.05      # шаг по времени (с)

# Создаём массивы для данных
time_steps = np.arange(0, t_max, dt)
y = np.zeros_like(time_steps)
v = np.zeros_like(time_steps)

# Начальные условия
y[0] = y0
v[0] = v0

# Метод Эйлера
for i in range(1, len(time_steps)):
    v[i] = v[i-1] - g * dt
    y[i] = y[i-1] + v[i-1] * dt

# Настройка графика
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-1, 1)
ax.set_ylim(0, y0 + 2)
ax.set_xlabel("x")
ax.set_ylabel("y (м)")
ax.set_title("Свободное падение (F = -mg)")
ax.grid(True)

# Объект для анимации (шарик)
ball, = ax.plot([0], [y0], 'o', markersize=20, color='blue')

# Функция анимации
def update(frame):
    ball.set_data([0], [y[frame]])
    return ball,

# Запуск анимации
ani = FuncAnimation(
    fig,
    update,
    frames=len(time_steps),
    interval=dt*1000,  # задержка между кадрами (мс)
    blit=False
)
plt.figure(figsize=(5, 5))
plt.plot( time_steps, y, color = "blue")
plt.plot(time_steps, v, color = "red")


plt.grid(True)
plt.show()
