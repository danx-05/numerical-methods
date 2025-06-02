# программа принимает фотографию, и возвращает размытую фотографию с заданными параметрами: kernel, sigma 
import math
import time
from numba import jit
import cv2
import numpy as np

def G(x,y,sigma):
    """
    Вычисление значения функции Гаусса при заданных x, y, sigma
    """
    return math.exp(-(x**2+y**2)/(2*sigma**2)) / (2 * math.pi * sigma ** 2 )
def gaussian_kernel(size, sigma):
    """
    Генерирует ядро Гаусса.
    Args:
        size: Размер ядра (должен быть нечетным).
        sigma: Стандартное отклонение.
    Returns:
        numpy.array: Ядро Гаусса.
    """
    """
        1  2  3 4 5 6 7
       -3 -2 -1 0 1 2 3
    """
    # создание
    matrix_kernel = np.ones((size, size))
    for x in range(-(size//2), size//2+1):
        for y in range(-(size//2), size//2+1):
            matrix_kernel[y + size // 2][x + size // 2] = G(x,y,sigma)
    sum_matrix_kernel = np.sum(matrix_kernel) # нахождение суммы элементов
    return matrix_kernel / sum_matrix_kernel # нормализация матрицы
@jit(nopython=True)
def blur(channel, matrix_kernel):
    """
    :param channel: Матрица одного из цветов
    :param matrix_kernel: Ядро
    :return: Новая матрица цвета
    """
    size = len(matrix_kernel)
    new_channel = np.copy(channel)
    # проходим по массиву ядром Гаусса
    for i in range(len(channel)):
        for j in range(len(channel[0]) ):
            summ = 0
            # вычисляем новое значение для [i][j] позиции
            for x in range(-(size//2), size//2+1):
                for y in range(-(size//2), size//2+1):
                    summ += channel[i + y][j + x] * matrix_kernel[y + size // 2][x + size // 2]
            new_channel[i][j] = summ

    return new_channel
def blur_Gauss(img, kernel_size, sigma):
    """
    :param img: Входящее изображение
    :param kernel_size: Размер ядра фильтра
    :param sigma: отклонение
    :return: профильтрованное изображение
    """
    matrix_kernel = gaussian_kernel(kernel_size, sigma) # вычисляем ядро Гаусса
    blue, green, red = cv2.split(img) # делим изоб-ние на три цвета

    # вычисляем для каждого цвета отдельно
    red_blurred = blur(red, matrix_kernel)
    green_blurred = blur(green, matrix_kernel)
    blue_blurred = blur(blue, matrix_kernel)

    blurred_image = cv2.merge([blue_blurred, green_blurred, red_blurred]) # объединяем цвета в изображение
    return np.clip(blurred_image, 0, 255) # делаем ограничение на значения от 0 до 255

if __name__ == '__main__':
    img = cv2.imread('kap.jpg')
    if img is None:
        print("Ошибка: Не удалось загрузить изображение.")
        exit()
    # Параметры размытия
    kernel_size = 25
    #sigma = 0.84089642
    sigma = 0.01

    # Применение размытия по Гауссу
    st = time.time()
    blurred_img = blur_Gauss(img, kernel_size, sigma)
    fin = time.time()
    print(f"{fin - st} секунд")

    cv2.imshow('Original image', img)
    cv2.imshow('Blurred image', blurred_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
