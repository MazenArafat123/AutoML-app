"""
pages/6_NLP_AutoML.py

NLP "Machine learning model" (Auto-ML) page: select a text column,
detect regression vs. classification from the target column, run
PyCaret's compare_models(), show the leaderboard, and let the user
download the best NLP model as a pickle file. Extracted from
final_project.py.

Relies on st.session_state.df / target_col / text_cols having been
set on the 'NLP Overview' page.
"""

import io

import numpy as np
import streamlit as st
from pycaret.classification import (
    setup as cls_setup,
    compare_models as cls_compare,
    pull as cls_pull,
    save_model as cls_save_model,
)
from pycaret.regression import (
    setup as reg_setup,
    compare_models as reg_compare,
    pull as reg_pull,
    save_model as reg_save_model,
)

from utils.state_utils import init_session_state

st.set_page_config(page_title="Auto-Machine learning app", layout="wide")

init_session_state()

st.title("NLP Models COMPARISON")

if st.session_state.df is None or st.session_state.target_col is None:
    st.info("Upload a dataset and select a target column first.")
else:
    df = st.session_state.df.copy()
    target_col = st.session_state.target_col
    text_cols = st.session_state.text_cols

    if not text_cols:
        st.warning("No text columns found. Go to Data Overview to set them up.")
    else:
        text_col = st.selectbox("Select text column for training", options=text_cols)

        y = df[target_col]

        # Detect task
        if np.issubdtype(y.dtype, np.number) and y.nunique() > 10:
            task = "regression"
        else:
            task = "classification"

        with st.expander("⏱️ Runtime settings", expanded=False):
            max_models = st.slider("Max models to compare", min_value=1, max_value=20, value=5, step=1)

        if st.button("🚀 Start Comparing (NLP)"):
            with st.spinner("Running PyCaret..."):
                nlp_df = df[[text_col, target_col]].copy()
                nlp_df = nlp_df.rename(columns={text_col: "text", target_col: "target"})

                if task == "classification":
                    cls_setup(data=nlp_df, target="target", session_id=42, verbose=False)
                    best_model = cls_compare(n_select=max_models)
                    leaderboard = cls_pull()
                else:
                    reg_setup(data=nlp_df, target="target", session_id=42, verbose=False)
                    best_model = reg_compare(n_select=max_models)
                    leaderboard = reg_pull()

            st.subheader("📊 NLP Model Leaderboard")
            st.dataframe(leaderboard)

            st.session_state.best_model_nlp = best_model

        if st.session_state.best_model_nlp is not None:
            buf = io.BytesIO()
            if task == "classification":
                cls_save_model(st.session_state.best_model_nlp, "best_nlp_model")
            else:
                reg_save_model(st.session_state.best_model_nlp, "best_nlp_model")

            with open("best_nlp_model.pkl", "rb") as f:
                st.download_button(
                    "📥 Download Best NLP Model",
                    f,
                    "best_nlp_model.pkl",
                    "application/octet-stream"
                )
