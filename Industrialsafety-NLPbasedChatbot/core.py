# -*- coding: utf-8 -*-
# - - - - - - - - - - - Sri Pandi - - - - - - - - - - - - - -

__author__ = 'Satheesh R'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


import json
import random
import re
import unicodedata
from string import punctuation

import pandas as pd
import streamlit as st
from nltk.corpus import stopwords

stop_words = stopwords.words('english')
import numpy as np
from bs4 import BeautifulSoup
from flair.data import Sentence
from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from scipy.spatial.distance import cosine
import os
from tensorflow.keras.utils import to_categorical

import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer

core_embeddings = DocumentPoolEmbeddings([WordEmbeddings('en')], pooling='mean')

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

pkl_file = os.path.join(BASE_DIR, 'assets', 'predicated_intents.pickle')
pre_file = os.path.join(BASE_DIR, 'assets', 'test_pred_answer.json')


def get_embeddings_(embedding_size):
    glove_file = f'./glove.6B.{embedding_size}d.txt'

    embeddings_ = {}
    for _line in open(glove_file, encoding='utf-8'):
        _word = _line.split(" ")[0]
        _embd = _line.split(" ")[1:]
        embeddings_[_word] = np.asarray(_embd, dtype='float32')
    return embeddings_


def prepare_embeddings(input_file, output_file):
    """

    :param input_file:
    :param output_file:
    :return:
    """

    embedded_intent_dict = {}
    with open(input_file) as file:
        intent_dict = json.load(file)

    for intent, examples in tqdm(intent_dict.items()):
        embedded_intent_dict[intent] = []
        for example in examples:
            sentence = Sentence(example)
            core_embeddings.embed(sentence)
            embedded_intent_dict[intent].append(sentence.embedding.detach().numpy())

    pickle.dump(embedded_intent_dict, open(os.path.join('assets', output_file), "wb+"))


def get_reply(msg, ebs, ans):
    """

    :param msg:
    :param ebs:
    :param ans:
    :return:
    """

    _intent, _score = '', 1

    with open(ebs, 'rb') as file:
        embedded_dict = pickle.load(file)

    message_sentence = Sentence(msg)
    core_embeddings.embed(message_sentence)
    message_vector = message_sentence.embedding.detach().numpy()

    for intent, replies in embedded_dict.items():
        for reply in replies:
            cur_score = cosine(message_vector, reply)
            if cur_score < _score:
                _score = cur_score
                _intent = intent

    with open(ans) as file:
        _answers = json.load(file)

    if _intent in _answers:
        return random.choice(_answers[_intent])
    else:
        return "I am sorry, Please rephrase your question and try again."


def get_response(input_message):
    """

    :param input_message:
    :return:
    """

    return str(get_reply(input_message, pkl_file, pre_file))


embeddings = DocumentPoolEmbeddings([WordEmbeddings('en')], pooling='mean', )


def process_the_data(df):
    """

    :param df:
    :return:
    """

    df_text = str(df.response).strip()

    if not isinstance(df_text, str):
        processed_text = np.nan
    else:
        print("------------------------------------------------")
        print("Original Text: \n", df_text)

        # Normalize with Unicode
        _text = unicodedata.normalize('NFKD', df_text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        #     ic(_text)

        _bs = BeautifulSoup(_text, "html.parser")
        [_str.extract() for _str in _bs(['iframe', 'script'])]
        _text = _bs.get_text()
        _text = re.sub(r'[\r|\n|\r\n]+', '\n', _text)

        # Remove special characters
        #         pattern = r'[^a-zA-z0-9\s]' #if not remove_digits else r'[^a-zA-z\s]'
        pattern = r"[^A-Za-z0-9!?\'\`]"
        _text = re.sub(pattern, ' ', _text)
        #     ic(_text)

        # Remove Stop words
        stop_words = stopwords.words('english') + list(punctuation) + ['urllink']
        _words = word_tokenize(_text)
        _words = [_word.lower() for _word in _words]
        _words = [_word for _word in _words if _word not in stop_words and not _word.isdigit()]
        words = list(dict.fromkeys(_words))
        #     ic(words)

        # Lemmatization
        _lemma = WordNetLemmatizer()
        _text = ' '.join([_lemma.lemmatize(_word) for _word in words])

        # Stemming
        _stemm = SnowballStemmer('english')
        _words = word_tokenize(_text)
        _stem_words = [_stemm.stem(_word) for _word in _words]
        processed_text = " ".join(_stem_words)
    #     print("\nProcessed Text: \n", processed_text)
    #     print("------------------------------------------------")

    df['Cleaned_Description'] = processed_text

    return df


def get_embeddings():
    """

    :return:
    """

    EMBEDDING_FILE = os.path.join(BASE_DIR, 'assets', 'glove.6B.200d.txt')
    _embeddings_index = {}
    with open(EMBEDDING_FILE, encoding="utf8") as f:
        for line in tqdm(f):
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            _embeddings_index[word] = coefs
    return _embeddings_index


# this function creates a normalized vector for the whole sentence
def sent2vec(s, e):
    words = str(s).lower()
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(e[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(300)
    return v / np.sqrt((v ** 2).sum())


def get_train_test(_df):
    """

    :return:
    """

    ind_tfidf_df = pd.DataFrame()
    for i in [1, 2, 3]:
        vec_tfidf = TfidfVectorizer(max_features=10, norm='l2', stop_words='english', lowercase=True, use_idf=True,
                                    ngram_range=(i, i))
        _x = vec_tfidf.fit_transform(_df['Cleaned_Description']).toarray()
        tfs = pd.DataFrame(_x, columns=["TFIDF_" + n for n in vec_tfidf.get_feature_names()])
        ind_tfidf_df = pd.concat([ind_tfidf_df.reset_index(drop=True), tfs.reset_index(drop=True)], axis=1)

    _df['Employee type'] = _df['Employee type'].str.replace(' ', '_')
    _df['Critical Risk'] = _df['Critical Risk'].str.replace('\n', '').str.replace(' ', '_')

    # Create Industry DataFrame
    ind_featenc_df = pd.DataFrame()

    # Label encoding
    _df['Season'] = _df['Season'].replace('Summer', 'aSummer').replace('Autumn', 'bAutumn').replace(
        'Winter', 'cWinter').replace('Spring', 'dSpring')
    ind_featenc_df['Season'] = LabelEncoder().fit_transform(_df['Season']).astype(np.int8)

    _df['Weekday'] = _df['Weekday'].replace('Monday', 'aMonday').replace('Tuesday', 'bTuesday'). \
        replace('Wednesday', 'cWednesday').replace('Thursday', 'dThursday').replace('Friday', 'eFriday'). \
        replace('Saturday', 'fSaturday').replace('Sunday', 'gSunday')
    ind_featenc_df['Weekday'] = LabelEncoder().fit_transform(_df['Weekday']).astype(np.int8)

    ind_featenc_df['Accident Level'] = LabelEncoder().fit_transform(_df['Accident Level']).astype(np.int8)
    ind_featenc_df['Potential Accident Level'] = LabelEncoder().fit_transform(
        _df['Potential Accident Level']).astype(np.int8)

    embeddings_index = get_embeddings()
    ind_glove_df = [sent2vec(x, embeddings_index) for x in tqdm(_df['Cleaned_Description'])]

    Country_dummies = pd.get_dummies(_df['Country'], columns=["Country"], drop_first=True)
    Local_dummies = pd.get_dummies(_df['Local'], columns=["Local"], drop_first=True)
    Gender_dummies = pd.get_dummies(_df['Gender'], columns=["Gender"], drop_first=True)
    IS_dummies = pd.get_dummies(_df['Industry Sector'], columns=['Industry Sector'], prefix='IS',
                                drop_first=True)
    EmpType_dummies = pd.get_dummies(_df['Employee type'], columns=['Employee type'], prefix='EmpType',
                                     drop_first=True)
    CR_dummies = pd.get_dummies(_df['Critical Risk'], columns=['Critical Risk'], prefix='CR', drop_first=True)

    # Merge the above dataframe with the original dataframe ind_feat_df
    ind_featenc_df = ind_featenc_df.join(Country_dummies.reset_index(drop=True)).join(
        Local_dummies.reset_index(drop=True)).join(Gender_dummies.reset_index(drop=True)).join(
        IS_dummies.reset_index(drop=True)).join(EmpType_dummies.reset_index(drop=True)).join(
        CR_dummies.reset_index(drop=True))

    ind_featenc_df = _df[['Year', 'Month', 'Day', 'WeekofYear']].reset_index(drop=True).join(
        ind_featenc_df.reset_index(drop=True))

    ind_feat_df = ind_featenc_df.join(pd.DataFrame(ind_glove_df).iloc[:, 0:30].reset_index(drop=True))
    ind_feat_df = ind_featenc_df.join(ind_tfidf_df.reset_index(drop=True))

    X = ind_feat_df.drop(['Accident Level', 'Potential Accident Level'], axis=1)  # Considering all Predictors
    y = ind_feat_df['Accident Level']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1, stratify=y)

    return X_train, X_test, y_train, y_test


def get_train_test_model(model, X_train, X_test, y_train, y_test, of_type='coef', save_model=True):
    """

    :param model:
    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :param of_type:
    :param save_model:
    :return:
    """

    model.fit(X_train, y_train)  # Fit the model on Training set

    if of_type == "coef":
        # Intercept and Coefficients
        print("The intercept for our model is {}".format(model.intercept_), "\n")

        for idx, col_name in enumerate(X_train.columns):
            print("The coefficient for {} is {}".format(col_name, model.coef_.ravel()[idx]))

    y_pred = model.predict(X_test)  # Predict on Test set

    train_accuracy_score = model.score(X_train, y_train)
    test_accuracy_score = model.score(X_test, y_test)

    st.info(f'Predicted Train Score:{train_accuracy_score}')
    st.info(f'Predicted Test Score:{test_accuracy_score}')

    p_score = precision_score(y_test, y_pred, average='weighted')
    r_score = recall_score(y_test, y_pred, average='weighted')
    f_score = f1_score(y_test, y_pred, average='weighted')

    st.info(f'Predicted Precision Score:{p_score}')
    st.info(f'Predicted Recall Score:{r_score}')
    st.info(f'Predicted F1 Score:{f_score}')

    # Save the model
    if save_model:
        file_name = os.path.join(BASE_DIR, 'data', 'logReg_model.sav')
        pickle.dump(model, open(file_name, 'wb'))
        st.success('Model saved successfully')
        st.success(file_name)
    return y_pred
