"""
utils/state_utils.py

Centralizes Streamlit session_state initialization for the app.
All keys mirror the original `init_session_state()` in final_project.py.
"""

import streamlit as st


def init_session_state():
    """Initialize all session_state keys used across the app, if not already set."""
    if "df" not in st.session_state:
        st.session_state.df = None
    if "orig_df" not in st.session_state:
        st.session_state.orig_df = None
    if "cleaned" not in st.session_state:
        st.session_state.cleaned = False
    if "cleaned_df" not in st.session_state:
        st.session_state.cleaned_df = None
    if "encoded_df" not in st.session_state:
        st.session_state.encoded_df = None
    if "target_col" not in st.session_state:
        st.session_state.target_col = None
    if "text_cols" not in st.session_state:
        st.session_state.text_cols = []
    if "best_model_tabular" not in st.session_state:
        st.session_state.best_model_tabular = None
    if "best_model_nlp" not in st.session_state:
        st.session_state.best_model_nlp = None
    if "vectorizer" not in st.session_state:
        st.session_state.vectorizer = None
