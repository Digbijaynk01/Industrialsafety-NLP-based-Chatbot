# -*- coding: utf-8 -*-
# - - - - - - - - - - - Sri Pandi - - - - - - - - - - - - - -

__author__ = 'Satheesh R'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import os
import re
# from core import process_the_data
import string

import pandas as pd
import streamlit as st
from core import get_train_test, get_train_test_model
from sklearn.linear_model import LogisticRegression

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

PUNCT_TO_REMOVE = string.punctuation

from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))


@st.cache
def load_data(_data):
    _df = pd.read_csv(_data)
    return _df


def run_pre_app():
    """

    :return:
    """

    st.subheader("Model Prediction.")
    submenu = st.sidebar.selectbox("SubMenu", ["Predict the Model", "Get Prediction"])
    data_set = load_data(os.path.join(BASE_DIR, 'data', 'Industrial_safety_df.csv'))

    if submenu.lower() == "predict the model":
        st.subheader("Get Train/test and predict the model.")

        # Building a Linear Regression model
        data_set.drop("Date", axis=1, inplace=True)
        lr = LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=1)

        X_train, X_test, y_train, y_test = get_train_test(_df=data_set)
        y_pred = get_train_test_model(lr, X_train, X_test, y_train, y_test )

        if y_pred:
            st.info('Model is ready for testing.')

    if submenu.lower() == "get prediction":
        st.subheader("Select models.")
