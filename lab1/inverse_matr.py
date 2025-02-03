def print_Matr(sp):
    for i in sp:
        print(*i)
def Gaus(a):
    """
        создаем матрицу и преобразуем (A|E) -> (E|A^(-1))
    """
    n = len(a)
    # единичная матрица nXn
    b = [n * [0] for _ in range(n)]
    for i in range(n):
        b[i][i] = 1
    # обычный прямой ход Гаусса
    for i in range(0, n):
        divisor = a[i][i]
        for k in range(len(a)):
            a[i][k] /= divisor
            b[i][k] /= divisor

        for j in range(i + 1, n):
            factor = a[j][i]
            for z in range(n):
                a[j][z] -= factor * a[i][z]
                b[j][z] -= factor * b[i][z]
    # метод обнуления над главной диагональю
    for i in range(n-1,0,-1): # идем по столбцам справа на лево
        for j in range(i-1,-1,-1): # идем по столбцу снизу вверх
            factor  = a[j][i]
            for k in range(n): # делим строки матрицы на a[j][i] и обнуляем a[j][i]
                a[j][k] -= factor * a[i][k]
                b[j][k] -= factor * b[i][k]
    return b
x = [[1,2,3,5],[1,4,3,7],[2,3,2,4], [3,2,1,3]]
print_Matr(Gaus(x))
