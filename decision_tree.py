import streamlit as st

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

from graphviz import Source


class DecisionTree:
    """Страница дерева решений."""

    def __init__(self, df):
        st.title('Дерево решений')
        self.df = df
        self.choose_features()
        self.model_fit()
        self.show_feature_importances()
        self.show_students_with_problems()
        self.show_metrics()

    def features_selection(self):
        """Выбор предикторов и целевой переменной."""
        target_feature = st.sidebar.selectbox(
            'Целевая переменная:', self.df.columns)

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
        max_depth = st.slider('Глубина дерева', 1, 6)

        self.best_tree = DecisionTreeClassifier(
            criterion='entropy',
            max_depth=max_depth,
            min_samples_leaf=1,
            min_samples_split=2
        )
        self.best_tree.fit(self.X_train, self.y_train)

        graph = Source(tree.export_graphviz(
            self.best_tree,
            out_file=None,
            feature_names=list(self.X),
            class_names=['Problem', 'No problem'],
            filled=True
        ))

        st.image(graph.pipe(format='png'))

    def show_feature_importances(self):
        """Отображение важности предикторов."""
        feature_importances = self.best_tree.feature_importances_
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

        self.predictions = self.best_tree.predict(self.X_test)
        students_with_problems = self.X_test.copy()
        students_with_problems['predictions'] = self.predictions
        st.code(
            students_with_problems[students_with_problems.iloc[:, -1] < .5])

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
