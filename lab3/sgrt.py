from decimal import *

def sqrt(a, precision):
    """
    :param a: корень числа a
    :param precision: точность для вычисления корня
    :return: корень числа a с заданной точностью
    """
    getcontext().prec = precision + 3 # увеличивает точность вычислений у Decimal
    a = Decimal(a) # преобразование a в Decimal
    x_0 = a
    e = Decimal("1e-" + str(precision + 1))
    # метод Ньютона
    x_1 = (x_0 + a / x_0) / 2
    while abs(x_1 - x_0) > e: # Условие остановки
        x_0 = x_1
        x_1 = (x_0 + a / x_0) / 2 # Нахождение следующего приближений
    return str(x_1.quantize(Decimal("1e-" + str(precision)))) # Округление до заданной точности


a = "2"
precision = 100
result = sqrt(a, precision)
print(result)
#s = "41421356237309504880168872420969807856967187537694807317667973799073247846210703885038753432764157273501384623091229702492483605585"


