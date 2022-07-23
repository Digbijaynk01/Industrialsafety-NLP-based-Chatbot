# -*- coding: utf-8 -*-
# - - - - - - - - - - - Sri Pandi - - - - - - - - - - - - - -

__author__ = 'Satheesh R'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import streamlit as st
import streamlit.components.v1 as stc

from chat_app import run_chat_app
from eda_app import run_eda_app
from init_app import run_init_app
from pre_app import run_pre_app

html_temp = """
<div style="background-color:rgba(194,193,193,0.23);padding:10px;border-radius:10px">
    <h1 style="color:#727272;text-align:center;">Industry - Safety : NLP ChatBot</h1>
    <h4 style="color:#727272;text-align:center;">Capstone-AIML</h4>
</div>
"""


def main():
    """
    ML Web App with Streamlit
    :return:
    """
    stc.html(html_temp)

    app_menu = ["Home", "Initialize", "Analysis", "Predictions", "ChatBot", "About"]
    menu_choice = st.sidebar.selectbox("MENU", app_menu)

    if menu_choice.lower() == "home":
        st.subheader("Home")
        st.write("""
        ### Industry - Safety : NLP ChatBot
        An AIML chatbot that recognizes the Accident Level, Potential Accident Level, 
        Critical Risks involved with the accident based on the descriptions provided.
        #### Datasource
        The database comes from one of the biggest industries in Brazil and in the world. 
        - https://archive.ics.uci.edu/ml/datasets/Early+stage+diabetes+risk+prediction+dataset.
        #### App Content
        - Home: Intro Page section/Home page
        - Initialize    : A new section/page for dataset loading and preprocessing.
        - Analysis      : A new section/page for Exploratory Data Analysis of Data.
        - Predictions   : A new section/page for moel prediction.
        - Chatbot       : A new section/page for a small chatbot.
        #### Authors
        - Amol / Digbijay / Satheesh / Vikram
        """)
    elif menu_choice.lower() == "initialize":
        run_init_app()
    elif menu_choice.lower() == "analysis":
        run_eda_app()
    elif menu_choice.lower() == "predictions":
        run_pre_app()
    elif menu_choice.lower() == "chatbot":
        run_chat_app()
    else:
        st.subheader("About")
        st.write("""
        ### Industry - Safety : NLP ChatBot
        A ML/DL based chatbot utility which can help the professional to determine the accident level 
        the accident level and potential level involved in any accident based on the text description
        """)
        st.text("Amol / Digbijay / Satheesh / Vikram")


if __name__ == '__main__':
    main()
