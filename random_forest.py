import streamlit as st

import base64
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split


class RandomForest:
    """Страница случайного леса."""

    def __init__(self, df):
        st.title('Cлучайный лес')
        self.df = df
        self.features_selection()
        self.model_fit()
        self.show_feature_importances()
        self.show_students_with_problems()
        self.show_metrics()

    def features_selection(self):
        """Выбор предикторов и целевой переменной."""
        target_feature = st.sidebar.selectbox(
            'Целевая переменная:', self.df.columns, len(self.df.columns) - 1)

        st.sidebar.text('Предикторы:')

        features = np.array(
            [f for f in self.df.columns if f != target_feature])
        selected_features = features[[
            st.sidebar.checkbox(f, f) for f in features]]

        self.X = self.df[selected_features]
        self.y = self.df[target_feature]
        self.X = pd.get_dummies(self.X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=.33)

    def model_fit(self):
        """Создание и обучение модели."""
        max_depth = st.sidebar.slider('Глубина деревьев', 1, 10, 4)
        n_estimators = st.sidebar.slider('Количество деревьев', 10, 300, 40)

        self.best_forest = RandomForestClassifier(
            max_depth=max_depth,
            min_samples_leaf=2, 
            min_samples_split=20,
            n_estimators=n_estimators
        )
        self.best_forest.fit(self.X_train, self.y_train)

    def show_feature_importances(self):
        """Отображение важности предикторов."""
        feature_importances = self.best_forest.feature_importances_
        feature_importances_df = pd.DataFrame({
            'feature_importances': feature_importances,
            'features': list(self.X_train)
        })

        st.subheader('Важность предикторов')
        st.write(feature_importances_df.sort_values(
            'feature_importances', ascending=False))

    def show_students_with_problems(self):
        """Отображение студентов в группе риска."""
        st.subheader('Выявленные студенты в группе риска.')

        self.predictions = self.best_forest.predict_proba(self.X_test)[:, 1]
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
