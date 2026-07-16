"""
pages/4_Tabular_AutoML.py

Tabular "Machine learning models" (Auto-ML) page: detects
regression vs. classification from the target column, runs PyCaret's
compare_models() across the preprocessed/cleaned/raw dataframe
(in that priority order), shows the resulting leaderboard, and lets
the user download the best model as a pickle file.
Extracted from final_project.py.

Relies on st.session_state.df / target_col / cleaned_df / encoded_df
having been set on the earlier Tabular pages.
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

st.title("Models comparison")

if st.session_state.df is None or st.session_state.target_col is None:
    st.info("Upload a dataset and select a target column first.")
else:
    if st.session_state.encoded_df is not None:
        df = st.session_state.encoded_df.copy()
    elif "cleaned_df" in st.session_state and st.session_state.cleaned_df is not None:
        df = st.session_state.cleaned_df.copy()
    else:
        df = st.session_state.df.copy()

    target_col = st.session_state.target_col

    if target_col not in df.columns:
        st.error("Selected target column is not in the dataframe.")
    else:
        y = df[target_col]

        # Detect task
        if np.issubdtype(y.dtype, np.number) and y.nunique() > 10:
            task = "regression"
        else:
            task = "classification"

        with st.expander("⏱️ Runtime settings", expanded=False):
            max_models = st.slider("Max models to compare", min_value=1, max_value=20, value=5, step=1)

        if st.button("🚀 Start Comparing (Tabular)"):
            with st.spinner("Running PyCaret..."):
                if task == "classification":
                    cls_setup(data=df, target=target_col, session_id=42, verbose=False)
                    best_model = cls_compare(n_select=max_models)
                    leaderboard = cls_pull()
                else:
                    reg_setup(data=df, target=target_col, session_id=42, verbose=False)
                    best_model = reg_compare(n_select=max_models)
                    leaderboard = reg_pull()

            st.subheader("📊 Model Leaderboard")
            st.dataframe(leaderboard)

            st.session_state.best_model_tabular = best_model

        if st.session_state.best_model_tabular is not None:
            buf = io.BytesIO()
            if task == "classification":
                cls_save_model(st.session_state.best_model_tabular, "best_tabular_model")
            else:
                reg_save_model(st.session_state.best_model_tabular, "best_tabular_model")

            with open("best_tabular_model.pkl", "rb") as f:
                st.download_button(
                    "📥 Download Best Tabular Model",
                    f,
                    "best_tabular_model.pkl",
                    "application/octet-stream"
                )
