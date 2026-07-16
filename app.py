"""
app.py

Main entry point for the Auto-Machine Learning Streamlit app.
Handles page config, session-state initialization, the file uploader,
and the target-column selector shared by every page.

Actual page content (Tabular EDA, Dashboard, Model Preprocessing,
Tabular AutoML, NLP Overview, NLP AutoML) lives in the pages/ folder;
Streamlit's native multipage navigation lists them automatically in
the sidebar.
"""

import seaborn as sns
import streamlit as st

from utils.state_utils import init_session_state
from utils.data_utils import load_file

# ---- Page config ----
st.set_page_config(page_title="Auto-Machine learning app", layout="wide")
sns.set_style("whitegrid")

# ---- Session state ----
init_session_state()

# ---- Landing page ----
st.title("🤖 Auto-Machine Learning App")
st.markdown(
    "Upload a dataset and pick a target column below, then use the pages "
    "listed in the sidebar to explore, clean, preprocess, and train models — "
    "either **Tabular** (*Tabular EDA → Dashboard → Model Preprocessing → "
    "Tabular AutoML*) or **NLP** (*NLP Overview → NLP AutoML*)."
)

# ---- Sidebar: file upload ----
st.sidebar.header("1. Upload data")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])
if uploaded_file is not None:
    with st.spinner("Loading file..."):
        df_loaded = load_file(uploaded_file)
        st.session_state.orig_df = df_loaded.copy()
        st.session_state.df = df_loaded.copy()
        st.success(f"Loaded '{uploaded_file.name}' ({df_loaded.shape[0]} rows × {df_loaded.shape[1]} cols).")

# ---- Sidebar: target selector ----
st.sidebar.header("2. Select target")
if st.session_state.df is not None:
    try:
        target_col = st.sidebar.selectbox(
            "Select target column",
            ["-- none --"] + st.session_state.df.columns.tolist()
        )
        st.session_state.target_col = None if target_col == "-- none --" else target_col
    except Exception:
        pass
else:
    st.sidebar.info("Upload a dataset to choose a target column.")

# ---- Landing page preview ----
if st.session_state.df is not None:
    st.subheader("Preview")
    st.dataframe(st.session_state.df.head())
    st.caption(f"Shape: {st.session_state.df.shape[0]} rows × {st.session_state.df.shape[1]} columns")
else:
    st.info("👈 Upload a CSV or Excel file in the sidebar to get started.")
