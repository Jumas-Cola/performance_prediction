import streamlit as st

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.tree import RandomForestClassifier

from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

from graphviz import Source

from dataframe import df


def decision_tree():
    """
    Создание, обучение и использование модели дерева решений.
    """

    st.title('Дерево решений')

    target_feature = st.sidebar.selectbox('Целевая переменная:', df.columns)

    st.sidebar.text('Предикторы:')
    features = np.array([f for f in df.columns if f != target_feature])
    selected_features = features[[st.sidebar.checkbox(f, f) for f in features]]

    X = df[selected_features]
    y = df[target_feature]

    X = pd.get_dummies(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33)

    max_depth = st.slider('Глубина дерева', 1, 6)

    best_tree = DecisionTreeClassifier(
        criterion='entropy',
        max_depth=max_depth,
        min_samples_leaf=1,
        min_samples_split=2
    )
    best_tree.fit(X_train, y_train)

    graph = Source(tree.export_graphviz(
        best_tree,
        out_file=None,
        feature_names=list(X),
        class_names=['Problem', 'No problem'],
        filled=True
    ))

    st.image(graph.pipe(format='png'))

    feature_importances = best_tree.feature_importances_
    feature_importances_df = pd.DataFrame({
        'feature_importances': feature_importances,
        'features': list(X_train)
    })
    st.write(feature_importances_df.sort_values('feature_importances', ascending=False))

    predictions = best_tree.predict(X_test)

    st.subheader('Выявленные студенты в группе риска.')

    students_with_problems = X_test.copy()
    students_with_problems['predictions'] = predictions
    st.code(students_with_problems[students_with_problems.iloc[:,-1] < .5])

    st.subheader('Метрики качества модели.')
    st.code(classification_report(y_test, predictions.round(), zero_division=True))

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

