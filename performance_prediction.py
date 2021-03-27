import streamlit as st

from functools import partial

from intro import intro
from data_processing import data_processing
from log_reg import LogReg
from decision_tree import DecisionTree
from random_forest import RandomForest

from dataframe import df 


def main():
    """
    Функция переключения страниц.
    """

    pages = {
        'Введение': intro,
        'Предварительный анализ': data_processing,
        'Логистическая регрессия': partial(LogReg, df),
        'Дерево решений': partial(DecisionTree, df),
        'Случайный лес': partial(RandomForest, df)
    }

    st.sidebar.title('Страницы')
    page = st.sidebar.radio('Выберите страницу', tuple(pages.keys()))

    # Отобразить выбранную страницу
    pages[page]()


if __name__ == '__main__':
    main()

