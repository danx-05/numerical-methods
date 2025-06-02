import random

from Lab10.Random import input_sp


def random_value(val, prob):
    """
    Задает случайную величину со значениями val и вероятностями prob,
    сумма prob должна быть равна 1
    :return: значение, зависимое от val и prob
    """
    func_distribution = [ sum(prob[0:i+1]) for i in range(len(prob))] # получили функцию распределения
    # теперь надо найти в какой промежуток попадет случайное число
    x = random.random()
    inter = 0
    for i in range(len(func_distribution)):
        if func_distribution[i] > x: # значение попало в промежуток
            inter = i
            break
    return val[inter] # возвращаем значение из промежутка


def my_random(seed, n):
    sp = [0] * n
    sp[0] = seed
    for i in range(1,n):
        sp[i] = sp[i-1] * 3 % 7
    return [ sp[i] % 6 + 1 for i in range(n)]

if __name__ == "__main__":
    probability = [0.1, 0.6, 0.1, 0.2]
    value = [10, 20, 30, 40]
    n = 10000
    sp = [ random_value(value, probability) for i in range(n)]
    input_sp(sp, value)
# ----------------------
#     n = 1000
#     seed = 5
#     sp = my_random(seed, n)
#     input_sp(sp, [x for x in range(1,7)] )
