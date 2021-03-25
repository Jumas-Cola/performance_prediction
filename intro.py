import streamlit as st

import numpy as np

import matplotlib.pyplot as plt


def intro():
    """
    Стартовая страница.
    """

    st.title('Введение')

    st.subheader('Идея проекта')

    st.write("""
    Целью каждого учителя является максимально полное усвоение учебной программы его учениками. 
    Для повышения качества обучения хорошей практикой было бы заблаговременное восполнение пробелов 
    в знаниях до того, как отставание станет значительным. Однако в условиях современной школы 
    достаточно трудно выявить учеников, которым необходимо уделить больше внимания и которым 
    может потребоваться более подробное объяснение материала.
    """)

    st.write("""
    Потенциальные трудности можно предусмотреть, основываясь на примерах из предыдущей практики. 
    Но на накопление достаточного багажа опыта могут потребоваться долгие годы, которых нет у 
    молодых специалистов. Поэтому было бы полезно использовать опыт других педагогов. Для этого 
    можно на основе данных об учениках и об их успеваемости построить модель, которая будет 
    предсказывать успеваемость по определённым факторам. Для соблюдения конфиденциальности, 
    имена и любая идентифицирующая информация может быть удалена из обучающей выборки.
    Важно отметить, что разрабатываемая система не может служить для стигматизации определённых 
    учеников, как «недостаточно успешных». Это лишь подсказка для педагога, которая поможет 
    скорректировать ход процесса обучения.
    """)

    st.subheader('Модель')

    st.write("""
    Для применения в проекте была выбрана одна из простейших моделей — логистическая регрессия, 
    уравнение которой задаёт многомерную плоскость, разделяющую точки наблюдений. На выходе 
    обученная модель возвращает вероятность принадлежности наблюдения к одному из двух классов. 
    В нашем случае: «успевающий» и «неуспевающий» ученик.
    """)

    st.write("""
    Логистическая регрессия или логит-модель (англ. logit model) — это статистическая модель, 
    используемая для прогнозирования вероятности возникновения некоторого события путём его 
    сравнения с логистической кривой. Эта регреcсия выдаёт ответ в виде вероятности бинарного 
    события (1 или 0). 
    """)

    x = np.arange(-6, 6, .1)
    y = (1 + np.exp(-x))**(-1)
    
    fig, ax = plt.subplots()
    plt.plot(x, y)
    st.pyplot(fig)

