# QR-алгоритм, тоже для нахождения собственных чисел
import numpy as np

def scar_mult(u, v):
    """
    Перемножает скалярно <u,v>
    :param u:
    :param v:
    :return:
    """
    summ = 0
    for i in range(len(v)):
        summ += u[i] * v[i]
    return summ
def modul(v):
    summ = 0
    for i in v:
        summ += i ** 2
    return summ ** 0.5
def qr_decomposition(A):
    """
    Реализация QR-разложения через процесс Грама-Шмидта.

    Параметры:
    A: квадратная матрица.

    Return:
    - Q: ортогональная матрица
    - R: верхняя треугольная матрица
    """
    n = A.shape[0]
    Q = np.zeros((n, n))  # ортогональная матрица
    R = np.zeros((n, n))  # верхняя треугольная матрица

    for j in range(n):  # Обрабатываем j-й столбец матрицы A
        v = A[:, j].copy()

        # Ортогонализация относительно предыдущих столбцов Q
        for i in range(j):
            R[i,j] = scar_mult(Q[:, i], A[:, j])
            v = v - R[i, j] * Q[:, i]

        # Нормализация
        R[j, j] = modul(v)
        Q[:, j] = v / R[j, j]

    return Q, R


def qr_algorithm_no_shift(A, max_iter=1000, eps=1e-10):
    """
    QR-алгоритм 
    Параметры:
    - A: квадратная матрица
    Возвращает:
    - eigenvalues: массив собственных значений
    """
    A_k = np.copy(A)

    for i in range(max_iter):
        Q, R = qr_decomposition(A_k)
        A_k = R @ Q
        print(i)
        print(A_k)
        print()
        # Проверка на сходимость
        diag = np.sum(np.abs(A_k - np.diag(np.diag(A_k))))
        if diag < eps:
            break

    eigenvalues = np.diag(A_k)
    return eigenvalues


if __name__ == "__main__":
    A = np.array([
        [5, 1, 0, 0, 0, 0, 0, 0],
        [1, 5, 1, 0, 0, 0, 0, 0],
        [0, 1, 5, 1, 0, 0, 0, 0],
        [0, 0, 1, 5, 1, 0, 0, 0],
        [0, 0, 0, 1, 5, 1, 0, 0],
        [0, 0, 0, 0, 1, 5, 1, 0],
        [0, 0, 0, 0, 0, 1, 5, 1],
        [0, 0, 0, 0, 0, 0, 1, 5]
    ], dtype=float)


    eigenvalues = qr_algorithm_no_shift(A)
    print(np.sort(eigenvalues))

    # Проверка через numpy
    exact_eigenvalues = np.linalg.eigvals(A)
    print(np.sort(exact_eigenvalues))
