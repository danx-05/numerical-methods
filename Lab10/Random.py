import random
import matplotlib.pyplot as plt


def math_wait(sp):
    return sum(sp) / len(sp)
def dispersion(sp):
    """
     d = 1/(n-1) * сумма (x_i - x_avr)
    """
    wait = math_wait(sp)
    disp = 0
    for i in range(len(sp)):
        disp += (sp[i] - wait) ** 2
    return disp / (len(sp) - 1)


def gistodram(data):
    # Сортируем данные по ключам для упорядоченного отображения
    sorted_items = sorted(data.items())
    keys = [str(k) for k, v in sorted_items]  # Преобразуем ключи в строки
    values = [v for k, v in sorted_items]

    total = sum(values)
    percentages = [(v / total) * 100 for v in values]

    # Создаем фигуру с настроенным размером
    fig, ax = plt.subplots(figsize=(12, 6))

    # Гистограмма с улучшенным отображением
    bars = ax.bar(keys, values, color='skyblue', edgecolor='black', width=0.7)


    # Добавляем проценты над столбцами
    for bar, percentage in zip(bars, percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.02 * max(values),
                f'{percentage:.1f}%\n({int(height)})',  # Добавляем и проценты и абсолютные значения
                 fontsize=10)



    plt.show()


def input_sp(sp, keys):
    d = {key : 0 for key in keys}
    for i in d.keys():
        d[i] = sp.count(i)
    print("Частоты: ", d)
    print("Мат. ожидание: ", math_wait(sp))
    print("Дисперсия: ", dispersion(sp))

    gistodram(d)


if __name__ == "__main__":
    n = 10  # Количество бросков

    # массив из n рандомных элементов
    sp = [ random.randint(1,6) for i in range(n)]
    input_sp(sp, [x for x in range(1,7)])




