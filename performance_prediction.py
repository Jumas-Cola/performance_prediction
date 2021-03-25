import streamlit as st

from intro import intro
from data_processing import data_processing
from logreg import logreg
from decision_tree import decision_tree

from dataframe import df 


def main():
    """
    Функция переключения страниц.
    """

    pages = {
        'Введение': intro,
        'Предварительный анализ': data_processing,
        'Логистическая регрессия': logreg,
        'Дерево решений': decision_tree
    }

    st.sidebar.title('Страницы')
    page = st.sidebar.radio('Выберите страницу', tuple(pages.keys()))

    # Отобразить выбранную страницу
    pages[page]()


if __name__ == '__main__':
    main()

