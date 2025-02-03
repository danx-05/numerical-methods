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
#x = [[2,2,3],[1,-1,0],[-1, 2, 1]]
#y = [1,0,2]
x = [
    [1,22,3],
    [21,21,1],
    [12,2,32]
]
y = [2,23,1]

print(Gaus(x,y))
