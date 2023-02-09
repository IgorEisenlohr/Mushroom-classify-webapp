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
    st.markdown("Are your mushrooms edible or poisonous? 🍄")
    st.sidebar.markdown("Are your mushrooms edible or poisonous? 🍄")

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
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input('C (Regularization parameter) ', 0.01, 10.0, step=0.01)
        kernel = st.sidebar.radio('Kernel', options=['rbf', 'linear'])
        gamma = st.sidebar.radio('Gamma (Kernel Coefficient)', options=['scale', 'auto'])

        metrics_list = st.sidebar.multiselect("What metrics to plot? ", options=['Confusion Matrix', 'ROC Curve', 'Precision Recall Curve'])
        if st.sidebar.button("Classify ", key='classify'):
            st.subheader('SVM model')
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            accuracy_train = model.score(X_train, y_train)
            accuracy_test  = model.score(X_test, y_test)

            #col1, col2 = st.columns(2)

            #with col1:
            st.write(f"Accuracy on train set: {round(accuracy_train, 2)}")
            st.write(f"Precision on train set: {round(precision_score(y_train_pred, y_train, labels=class_names), 2)}")
            st.write(f"Recall on train set: {round(recall_score(y_train_pred, y_train, labels=class_names), 2)}")

            #with col2:
            st.write(f"Accuracy on test set:  {round(accuracy_test, 2)}")
            st.write(f"Precision on test set:  {round(precision_score(y_test_pred, y_test, labels=class_names), 2)}")           
            st.write(f"Accuracy on test set:  {round(recall_score(y_test_pred, y_test, labels=class_names), 2)}")

            plot_metrics(metrics_list)

    if classifier == 'Logistic Regression':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input('C (Regularization parameter) ', 0.01, 10.0, step=0.01)
        max_iter = st.sidebar.slider('Max number of iterations', 100, 1000)

        metrics_list = st.sidebar.multiselect("What metrics to plot? ", options=['Confusion Matrix', 'ROC Curve', 'Precision Recall Curve'])
        if st.sidebar.button("Classify ", key='classify2'):
            st.subheader('Logistic Regression model')
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            accuracy_train = model.score(X_train, y_train)
            accuracy_test  = model.score(X_test, y_test)

            #col1, col2 = st.columns(2)

            #with col1:
            st.write(f"Accuracy on train set: {round(accuracy_train, 2)}")
            st.write(f"Precision on train set: {round(precision_score(y_train_pred, y_train, labels=class_names), 2)}")
            st.write(f"Recall on train set: {round(recall_score(y_train_pred, y_train, labels=class_names), 2)}")

            #with col2:
            st.write(f"Accuracy on test set:  {round(accuracy_test, 2)}")
            st.write(f"Precision on test set:  {round(precision_score(y_test_pred, y_test, labels=class_names), 2)}")           
            st.write(f"Accuracy on test set:  {round(recall_score(y_test_pred, y_test, labels=class_names), 2)}")

            plot_metrics(metrics_list)

    if classifier == 'Random Forest':
        st.sidebar.subheader("Model Hyperparameters")
        bootstrap = st.sidebar.radio('Bootstrap: ',['True','False'])
        max_depth = st.sidebar.number_input('Max depth ', 1, 20, step=1)
        n_estimators = st.sidebar.slider('N estimators', 50, 250)

        metrics_list = st.sidebar.multiselect("What metrics to plot? ", options=['Confusion Matrix', 'ROC Curve', 'Precision Recall Curve'])
        if st.sidebar.button("Classify ", key='classify2'):
            st.subheader('Random Forest model')
            model = RandomForestClassifier(bootstrap=bootstrap, max_depth=max_depth, n_estimators=n_estimators)
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            accuracy_train = model.score(X_train, y_train)
            accuracy_test  = model.score(X_test, y_test)

            #col1, col2 = st.columns(2)

            #with col1:
            st.write(f"Accuracy on train set: {round(accuracy_train, 2)}")
            st.write(f"Precision on train set: {round(precision_score(y_train_pred, y_train, labels=class_names), 2)}")
            st.write(f"Recall on train set: {round(recall_score(y_train_pred, y_train, labels=class_names), 2)}")

            #with col2:
            st.write(f"Accuracy on test set:  {round(accuracy_test, 2)}")
            st.write(f"Precision on test set:  {round(precision_score(y_test_pred, y_test, labels=class_names), 2)}")           
            st.write(f"Accuracy on test set:  {round(recall_score(y_test_pred, y_test, labels=class_names), 2)}")

            plot_metrics(metrics_list)

        if st.sidebar.checkbox("Show raw data", False):
            st.subheader("Mushroom Data Set (Classification)")
            st.write(df)

if __name__ == '__main__':
    main()