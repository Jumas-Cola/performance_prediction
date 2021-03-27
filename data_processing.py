import streamlit as st

import math
import itertools as it
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
    rows = math.ceil(len(df.columns) / 2)
    cols = 2
    fig, ax = plt.subplots(rows, cols, figsize=(4, 1.5*rows))
    fig.tight_layout(h_pad=2.5)

    coord_pairs = it.product(range(rows), range(cols))
    for i, coords in enumerate(coord_pairs):
        r, c = coords
        ax[r, c].xaxis.set_visible(True)
        for tick in ax[r, c].get_xticklabels():
            tick.set_rotation(25)
        sns.histplot(data=df, x=df.columns[i], ax=ax[r, c])
        if i + 1 == len(df.columns):
            coords = next(coord_pairs, None)
            if coords:
                r, c = coords
                ax[r, c].axis('off')
            break
    st.pyplot(fig)

