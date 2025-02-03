def print_Matr(sp):
    for i in sp:
        print(*i)
def Gaus(x, y):
    n = len(y) # n размерность матрицы
    ans = [0] * n
    # прямой ход Гаусса
    for i in range(0, n):
        divisor = x[i][i] # элемент главной диагонали
        # цикл деления строки марицы, чтобы на позиции [i][i] стояла 1
        for k in range(len(x)):
            x[i][k] /= divisor
        # и свободный член
        y[i] /= divisor
        # обнуление столбца под позицией [i][i]
        for j in range(i + 1, n):
            factor = x[j][i]
            for z in range(n):
                x[j][z] -= factor * x[i][z]
            y[j] -= factor * y[i]
    print_Matr(x)
    # обратный ход Гаусса
    for i in range(n - 1, -1, -1):
        # получение ответа
        ans[i] = y[i] / x[i][i]
        for j in range(0, i):
            y[j] -= x[j][i] * ans[i]
    return ans
x = [[1-1j, 2],
     [3+1j, 4+1j]
    ]
y = [2, 5+6j]
print(Gaus(x,y))
