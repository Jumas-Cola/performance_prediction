import streamlit as st

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from dataframe import df


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


