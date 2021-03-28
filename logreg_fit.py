import streamlit as st

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import statsmodels.api as sm 
from statsmodels.formula.api import glm

from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

from dataframe import df


def logreg_fit():
    """
    Создание, обучение и использование модели логистической регрессии.
    """

    st.title('Создание и использование модели логистической регрессии')

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


