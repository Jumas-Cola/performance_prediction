import streamlit as st

import pandas as pd


@st.cache
def load_data(path):
    data = pd.read_csv(path)
    return data


df = load_data('data/StudentsPerformance.csv')
