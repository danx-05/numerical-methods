import cv2
import numpy as np
from numba import jit

@jit(nopython=True)
def gray_image(blue, green, red):
    for i in range(len(blue)):
        for j in range(len(blue[0])):
            gray = 0.299 * red[i][j] + 0.587 * green[i][j] + 0.114 * blue[i][j]
            blue[i][j] = green[i][j] = red[i][j] = gray
    return blue, green, red

@jit(nopython=True)
def filtering(channel, kernel):
    size = len(kernel)
    new_channel = np.zeros_like(channel, dtype=np.float32)
    for i in range(size//2, len(channel)-size//2):
        for j in range(size//2, len(channel[0])-size//2):
            summ = 0.0
            for x in range(-(size//2), size//2+1):
                for y in range(-(size//2), size//2+1):
                    summ += channel[i + y][j + x] * kernel[y + size//2][x + size//2]
            new_channel[i][j] = summ
    return new_channel


def normalize_image(src, alpha=0, beta=255, dtype=np.uint8):
    """
    Нормализует массив изображения в заданный диапазон [alpha, beta].
    Аналог cv2.normalize(src, None, alpha, beta, cv2.NORM_MINMAX, dtype).

    Параметры:
        src: входной массив (numpy.ndarray)
        alpha: новый минимальный диапазон (по умолчанию 0)
        beta: новый максимальный диапазон (по умолчанию 255)
        dtype: тип данных выходного массива (по умолчанию np.uint8)

    Возвращает:
        Нормализованный массив в диапазоне [alpha, beta]
    """
    # Находим минимальное и максимальное значения в массиве
    src_min = np.min(src)
    src_max = np.max(src)
    # Линейное преобразование значений
    normalized = (src - src_min) * ((beta - alpha) / (src_max - src_min)) + alpha
    return normalized.astype(dtype)
def canny_detector(img, output_filename):
    blue, green, red = cv2.split(img)
    blue, green, red = gray_image(blue, green, red)

    # Ядра Собеля
    Sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)
    Sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=np.float32)

    Gx = filtering(blue, Sobel_x)
    Gy = filtering(blue, Sobel_y)
    G_new = np.zeros((len(Gx), len(Gx[0])), dtype=np.float32)
    # Магнитуда градиента (без переполнения)
    for i in range(len(Gx)):
        for j in range(len(Gx[0])):
            G_new[i][j] = np.sqrt(Gx[i][j]**2 + Gy[i][j]**2)

    # Нормализация для отображения
    Gx_vis = normalize_image(Gx)
    Gy_vis = normalize_image(Gy)

    # Нормализация и сохранение
    result_image = normalize_image(G_new)
    cv2.imwrite(output_filename, result_image)

    cv2.imshow('Sobel X', Gx_vis)
    cv2.imshow('Sobel Y', Gy_vis)

    cv2.waitKey(0)
    return result_image

if __name__ == "__main__":
    img = cv2.imread('kap.jpg')
    if img is None:
        print("Ошибка: Не удалось загрузить изображение.")
        exit()
    edges = canny_detector(img, 'edge_kap.png')
