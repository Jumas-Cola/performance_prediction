import streamlit as st

import pandas as pd


@st.cache
def load_data(path, *args, **kwargs):
    data = pd.read_csv(path, *args, **kwargs)
    return data


#df = load_data('data/StudentsPerformance.csv')
#df = load_data('data/student/student-mat.csv', sep=';')
df = load_data('data/student/student-por.csv', sep=';')
