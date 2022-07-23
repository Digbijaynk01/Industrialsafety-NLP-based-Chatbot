# -*- coding: utf-8 -*-
# - - - - - - - - - - - Sri Pandi - - - - - - - - - - - - - -

__author__ = 'Satheesh R'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


import streamlit as st

# from chatterbot import ChatBot
import core as bot_obj


def get_text():
    input_text = st.text_input("You: ", '')
    return input_text


def run_chat_app():
    # bot_obj = ChatBot(name='PyBot', read_only=False,
    #                   preprocessors=['chatterbot.preprocessors.clean_whitespace',
    #                                  'chatterbot.preprocessors.convert_to_ascii',
    #                                  'chatterbot.preprocessors.unescape_html'],
    #                   logic_adapters=['chatterbot.logic.MathematicalEvaluation',
    #                                   'chatterbot.logic.BestMatch'])

    user_input = get_text()

    if True:
        if user_input:
            st.text_area("Bot:", value=bot_obj.get_response(user_input), height=200, max_chars=None, key=None)
        else:
            st.text_area("Bot:", value='Hi, I can help you on Industrial Safety', height=200, max_chars=None, key=None)
    else:
        st.text_area("Bot:", value="Please start the bot by clicking sidebar button", height=200,
                     max_chars=None, key=None)
