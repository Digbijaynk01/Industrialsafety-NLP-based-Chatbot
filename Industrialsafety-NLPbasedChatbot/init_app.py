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

PUNCT_TO_REMOVE = string.punctuation
import unicodedata

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

STOPWORDS = set(stopwords.words('english'))
BASE_DIR = os.path.abspath(os.path.dirname(__file__))


def month2seasons(x):
    if x in [9, 10, 11]:
        season = 'Spring'
    elif x in [3, 4, 5]:
        season = 'Autumn'
    elif x in [6, 7, 8]:
        season = 'Winter'
    else:
        season = 'Summer'
    return season

def remove_stopwords(text):
    """

    :param text:
    :return:
    """
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])


def replace_words(df_text):
    """

    :param df_text:
    :return:
    """

    _text = unicodedata.normalize('NFKD', df_text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    _bs = BeautifulSoup(_text, "html.parser")
    [_str.extract() for _str in _bs(['iframe', 'script'])]
    _text = _bs.get_text()
    _text = re.sub(r'[\r|\n|\r\n]+', '\n', _text)
    return _text


def remove_punctuation(text):
    """

    :param text:
    :return:
    """
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))


lemmatizer = WordNetLemmatizer()


def lemmatize_words(text):
    """

    :param text:
    :return:
    """
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])


@st.cache
def load_data(_data):
    _df = pd.read_csv(_data)
    return _df



def run_init_app():
    """

    :return:
    """

    st.subheader("Preprocessing.")
    submenu = st.sidebar.selectbox("SubMenu", ["Load the Dataset", "Preprocess the Dataset"])

    if submenu.lower() == "load the dataset":
        st.subheader("Upload the Dataset for analysis.")

        file_details = ''
        st.subheader("Dataset")
        data_file = st.file_uploader("Upload CSV", type=["csv"])

        if data_file is not None:
            file_details = {"filename": data_file.name, "filetype": data_file.type, "filesize": data_file.size}
            st.write(file_details)

            df = load_data(data_file)
            st.dataframe(df)

            if not df.empty:
                df.to_csv(os.path.join(BASE_DIR, 'data', 'Industrial_safety.csv'))
                st.success("File uploaded successfully")
            else:
                st.error("Failed to upload")

    if submenu.lower() == "preprocess the dataset":
        st.subheader("Upload the Dataset for analysis.")

        df_orig = load_data(os.path.join(BASE_DIR, 'data', 'Industrial_safety.csv'))
        st.dataframe(df_orig.head())

        df_orig.drop("Unnamed: 0", axis=1, inplace=True, errors='ignore')
        df_orig.drop("Unnamed: 0.1", axis=1, inplace=True, errors='ignore')
        df_orig.rename(columns={'Data': 'Date', 'Countries': 'Country', 'Genre': 'Gender',
                                'Employee or Third Party': 'Employee type'}, inplace=True)
        df_orig.drop_duplicates(inplace=True)

        # df_processed = df_orig.progress_apply(process_the_data, axis=1)
        # df_processed = process_the_data(df_orig)

        print('--' * 30)
        st.info('Converting description to lower case')
        df_orig['Cleaned_Description'] = df_orig['Description'].apply(lambda x: x.lower())

        st.info('Replacing apostrophes to the standard lexicons')
        df_orig['Cleaned_Description'] = df_orig['Cleaned_Description'].apply(lambda x: replace_words(x))

        st.info('Removing punctuations')
        df_orig['Cleaned_Description'] = df_orig['Cleaned_Description'].apply(lambda x: remove_punctuation(x))

        st.info('Applying Lemmatizer')
        df_orig['Cleaned_Description'] = df_orig['Cleaned_Description'].apply(lambda x: lemmatize_words(x))

        st.info('Removing multiple spaces between words')
        df_orig['Cleaned_Description'] = df_orig['Cleaned_Description'].apply(lambda x: re.sub(' +', ' ', x))

        st.info('Removing stop words')
        df_orig['Cleaned_Description'] = df_orig['Cleaned_Description'].apply(lambda x: remove_stopwords(x))

        st.info('Converting month to season')
        df_orig['Date'] = pd.to_datetime(df_orig['Date'])
        df_orig['Year'] = df_orig.Date.apply(lambda x: x.year)
        df_orig['Month'] = df_orig.Date.apply(lambda x: x.month)
        df_orig['Day'] = df_orig.Date.apply(lambda x: x.day)
        df_orig['Weekday'] = df_orig.Date.apply(lambda x: x.day_name())
        df_orig['WeekofYear'] = df_orig.Date.apply(lambda x: x.weekofyear)

        df_orig['Season'] = df_orig['Month'].apply(lambda x: month2seasons(x))
        df_orig.to_csv(os.path.join(BASE_DIR, 'data', 'Industrial_safety_df.csv'), index=False)
        # st.dataframe(df_orig)
        st.success("Data preprocessed/cleaned successfully -")
        st.info(os.path.join(BASE_DIR, 'data', 'Industrial_safety_df.csv'))
