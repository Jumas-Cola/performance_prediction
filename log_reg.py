import streamlit as st

import base64
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import statsmodels.api as sm 
from statsmodels.formula.api import glm

from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split


class LogReg:
    """Страница логистической регрессии."""

    def __init__(self, df):
        st.title('Логистическая регрессиия')
        self.df = df
        self.features_selection()
        self.model_fit()
        self.show_summary()
        self.show_students_with_problems()
        self.show_metrics()

    def features_selection(self):
        """Выбор предикторов и целевой переменной."""
        self.target_feature = st.sidebar.selectbox('Целевая переменная:', self.df.columns, len(self.df.columns) - 1)

        st.sidebar.text('Предикторы:')
        features = np.array([f for f in self.df.columns if f != self.target_feature])
        self.selected_features = features[[st.sidebar.checkbox(f, f) for f in features]]

        self.X = self.df[self.selected_features]
        self.y = self.df[self.target_feature]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=.33)

    def model_fit(self):
        """Создание и обучение модели."""
        self.log_reg = glm(
            f'{self.target_feature} ~ {" + ".join(self.selected_features) if len(self.selected_features) else "1"}',
            data=pd.concat([self.X_train, self.y_train], axis=1),
            family=sm.families.Binomial()).fit()

    def show_summary(self):
        """Сводная таблица логистической регрессиии."""
        st.subheader('Сводная таблица.')
        st.code(self.log_reg.summary())

    def show_students_with_problems(self):
        """Отображение учеников в группе риска."""
        st.subheader('Выявленные ученики в группе риска.')

        self.predictions = self.log_reg.predict(self.X_test)
        students_with_problems = self.X_test.copy()
        students_with_problems['predictions'] = self.predictions
        threshold = st.slider('Пороговое значение:', .01, .99, .5)
        students_with_problems = students_with_problems[students_with_problems.iloc[:,-1] < threshold]
        st.write(f'Всего: {students_with_problems.shape[0]}')
        st.write(students_with_problems)

        csv = students_with_problems.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="students_with_problems.csv">Download CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

    def show_metrics(self):
        """Отображение метрик качества модели."""
        st.subheader('Метрики качества модели.')
        st.code(classification_report(
            self.y_test, self.predictions.round(), zero_division=True))

        fpr, tpr, thresholds = roc_curve(self.y_test, self.predictions)
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

