"""
pages/3_Model_Preprocessing.py

Tabular "Model preprocessing" page: normalize/scale numeric columns,
encode categorical columns (One-Hot / Label / Mapping), and download
the resulting preprocessed dataset as CSV. Extracted from final_project.py.

Relies on st.session_state.df / cleaned_df having been set on the
'Tabular EDA' page.
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

from utils.state_utils import init_session_state
from utils.data_utils import normalize_columns_inplace

st.set_page_config(page_title="Auto-Machine learning app", layout="wide")

init_session_state()

st.title("Model Preprocessing")

if st.session_state.df is None:
    st.info("Upload and clean a dataset first.")
else:
    active_df = (
        st.session_state.cleaned_df.copy()
        if "cleaned_df" in st.session_state and st.session_state.cleaned_df is not None
        else st.session_state.df.copy()
    )

    # Normalization / Scaling
    st.markdown("### 📏 Normalize Columns")
    numeric_cols = active_df.select_dtypes(include=np.number).columns.tolist()
    cols_norm = st.multiselect("Select numeric columns to normalize", options=numeric_cols)
    if st.button("Normalize Selected Columns"):
        if cols_norm:
            with st.spinner("Normalizing..."):
                normalize_columns_inplace(cols_norm)
            st.success(f"Normalized: {cols_norm}")
            st.rerun()
        else:
            st.warning("No numeric columns selected.")

    # Encoding
    st.markdown("### 🔡 Encoding Options")
    cat_small = [
        c for c in active_df.select_dtypes(include=["object", "category"]).columns
        if active_df[c].nunique() < 15
    ]

    if not cat_small:
        st.info("No categorical features to encode")
    else:
        if 'encoded_df' not in st.session_state or st.session_state.encoded_df is None:
            st.session_state.encoded_df = active_df.copy()

        for col in cat_small:
            with st.expander(f"🔧 Encode '{col}'"):
                st.markdown(f"**Values:** {active_df[col].unique().tolist()}")
                method = st.selectbox(
                    f"Encoding method for {col}",
                    ["One-Hot", "Label", "Mapping"],
                    key=f"enc_method_{col}"
                )

                mapping = None
                if method == "Mapping":
                    mapping = {}
                    for val in active_df[col].unique():
                        new_val = st.text_input(f"Map '{val}' →", val, key=f"map_{col}_{val}")
                        mapping[val] = new_val

                if st.button(f"Apply {method} Encoding to '{col}'", key=f"btn_enc_{col}"):
                    df_enc = st.session_state.encoded_df.copy()
                    if method == "One-Hot":
                        df_enc = pd.get_dummies(df_enc, columns=[col], drop_first=True)
                    elif method == "Label":
                        le = LabelEncoder()
                        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
                    elif method == "Mapping":
                        df_enc[col] = df_enc[col].map(mapping)

                    st.session_state.encoded_df = df_enc
                    st.success(f"✅ Applied {method} encoding to '{col}'")

    if 'encoded_df' in st.session_state and st.session_state.encoded_df is not None:
        csv = st.session_state.encoded_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Preprocessed Dataset", csv, "preprocessed.csv", "text/csv")
