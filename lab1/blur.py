import math
import time
from numba import jit, prange, njit
import cv2
import numpy as np

def G(x,y,sigma):
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
    matrix_kernel = np.ones((size, size))
    for x in range(-(size//2), size//2+1):
        for y in range(-(size//2), size//2+1):
            matrix_kernel[y + size // 2][x + size // 2] = G(x,y,sigma)
    return matrix_kernel
@jit(nopython=True)
def blur(channel, matrix_kernel):
    """
    :param channel: Матрица одного из цветов
    :param matrix_kernel: Ядро
    :return: Новая матрица цвета
    """
    size = len(matrix_kernel)
    new_channel = np.copy(channel)
    for i in range(len(channel)):
        for j in range(len(channel[0]) ):
            summ = 0
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
    matrix_kernel = gaussian_kernel(kernel_size, sigma)
    print(matrix_kernel)
    blue, green, red = cv2.split(img)

    red_blurred = blur(red, matrix_kernel)
    green_blurred = blur(green, matrix_kernel)
    blue_blurred = blur(blue, matrix_kernel)

    blurred_image = cv2.merge([blue_blurred, green_blurred, red_blurred])
    return np.clip(blurred_image, 0, 255)

if __name__ == '__main__':
    img = cv2.imread('kap.jpg')

    if img is None:
        print("Ошибка: Не удалось загрузить изображение.")
        exit()

    # Параметры размытия
    kernel_size = 5
    sigma = 0.84089642

    # Применение размытия по Гауссу
    st = time.time()
    blurred_img = blur_Gauss(img, kernel_size, sigma)
    fin = time.time()
    print(f"{fin - st} секунд")

    cv2.imshow('Original image', img)
    cv2.imshow('Blurred image', blurred_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
