import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve, confusion_matrix
from sklearn.metrics import precision_score, recall_score 

def main():
    st.title('Binary classification Web App')
    st.sidebar.title('Binary classification Web App')
    st.markdown("Are your mushrooms edible or poisonous? Ã°ÂÂÂ")
    st.sidebar.markdown("Are your mushrooms edible or poisonous? Ã°ÂÂÂ")

    @st.cache(persist=True)
    def load_data():
        df = pd.read_csv('mushrooms.csv')
        for col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
        return df                

    @st.cache(persist=True)
    def split_data(df):
        X = df.drop(columns=['type'])
        y = df.type

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        return X_train, X_test, y_train, y_test

    def plot_metrics(metrics_list):
        st.set_option('deprecation.showPyplotGlobalUse', False)
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test_pred, y_test)
            sns.heatmap(cm, annot=True, fmt='.1f')
            st.pyplot()
        
        if 'ROC Curve' in metrics_list:       
            st.subheader("ROC Curve")
            plot_roc_curve(model, X_test, y_test)
            st.pyplot()

        if 'Precision Recall Curve' in metrics_list:
            st.subheader('Precision Recal Curve')
            plot_precision_recall_curve(model, X_test, y_test)
            st.pyplot()

    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    class_names = ['edible', 'poisonous']
    
    st.sidebar.subheader('Choose your classifier')
    classifier = st.sidebar.selectbox("Classifier", options=['SVM', 'Logistic Regression', 'Random Forest'])

    if classifier == 'SVM':
        st.sidebar.subheader("Model H
