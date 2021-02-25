import streamlit as st

import math
import random as rd
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

import statsmodels.api as sm 
from statsmodels.formula.api import glm

from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split


df = pd.read_csv('data/StudentsPerformance.csv')

def main():
    """
    Функция переключения страниц.
    """

    pages = {
        'Введение': intro,
        'Предварительный анализ': data_processing,
        'Создание и использование модели': model_fit
    }

    st.sidebar.title('Страницы')
    page = st.sidebar.radio('Выберите страницу', tuple(pages.keys()))

    # Отобразить выбранную страницу
    pages[page]()


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


def data_processing():
    """
    Страница предварительного анализа исходных данных.
    """

    st.title('Предварительный анализ')

    st.write(df.head(10))

    # Отрисовка гистограмм столбцов исходных данных
    fig, ax = plt.subplots(3, 2)
    fig.tight_layout()
    ax[2, 0].axis('off')

    sns.histplot(data=df, x='gender', ax=ax[0, 0])
    sns.histplot(data=df, x='ethnicity', ax=ax[0, 1])
    sns.histplot(data=df, x='parental_level_of_education', ax=ax[1, 0])

    ax[1, 0].xaxis.set_visible(True)
    for tick in ax[1, 0].get_xticklabels():
        tick.set_rotation(90)

    sns.countplot(data=df, x='lunch', ax=ax[1, 1])
    sns.countplot(data=df, x='test_preparation_course', ax=ax[2, 1])
    st.pyplot(fig)


    # Построение точечного 3D графика для учебных предметов
    fig = px.scatter_3d(df, x='math_score', y='rus_score', z='literature_score',
                  color='have_no_problem', height=600)
    st.plotly_chart(fig)


    # Построение попарных точечных графиков и гистограмм для отдельных учебных предметов
    ax = sns.pairplot(df[['math_score', 'literature_score', 'rus_score']])
    st.pyplot(ax)


def model_fit():
    """
    Создание, обучение и использование модели логистической регрессии.
    """

    st.title('Создание и использование модели')

    st.sidebar.header('Предикторы:')

    features = np.array(['gender', 'ethnicity', 'lunch', 'test_preparation_course', 'parental_level_of_education'])
    selected_features = features[[st.sidebar.checkbox(f, f) for f in features]]

    X = df[features]
    y = df['have_no_problem']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33)

    log_reg = glm(
        f'have_no_problem ~ {" + ".join(selected_features) if len(selected_features) else "1"}',
        data=pd.concat([X_train, y_train], axis=1),
        family=sm.families.Binomial()).fit()

    st.subheader('Сводная таблица.')

    st.code(log_reg.summary())

    predictions = log_reg.predict(X_test)

    st.subheader('Выявленные студенты в группе риска.')

    students_with_problems = X_test
    students_with_problems['predictions'] = predictions
    st.code(students_with_problems[students_with_problems.iloc[:,-1] < .5])

    st.subheader('Метрики качества модели.')
    st.code(classification_report(y_test, round(predictions)))

    fpr, tpr, thresholds = roc_curve(y_test, predictions)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    st.pyplot(plt)


if __name__ == '__main__':
    main()

