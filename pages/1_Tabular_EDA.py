"""
pages/1_Tabular_EDA.py

Tabular "EDA & cleaning" page: data preview, dataset info, missing
values, descriptive stats, the Clean & Analyze button, column dropping,
and the cleaned-data preview. Extracted from final_project.py.

Relies on st.session_state.df / target_col having been set on the
main app.py page (file upload + target selection happen there).
"""

from io import StringIO

import streamlit as st

from utils.state_utils import init_session_state
from utils.data_utils import clean_and_analyze_normal

st.set_page_config(page_title="Auto-Machine learning app", layout="wide")

init_session_state()

st.title("📁 Data & cleaning")

if st.session_state.df is None:
    st.info("Upload a CSV / Excel file from the left sidebar to begin.")
else:
    df = st.session_state.df
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())

    st.subheader("📏 Dataset Info")
    buf = StringIO()
    df.info(buf=buf)
    st.text(buf.getvalue())

    st.subheader("📉 Missing Values")
    missing = df.isnull().sum()
    if missing[missing > 0].empty:
        st.write("No missing values.")
    else:
        st.dataframe(missing[missing > 0])

    st.subheader("📊 Descriptive Statistics")
    st.write(df.describe(include="all"))

    st.markdown("---")
    st.header("⚙️ Final Processing")

    if st.button("Clean & Analyze Data"):
        with st.spinner("Cleaning..."):
            clean_and_analyze_normal()
        st.session_state.cleaned = True
        st.rerun()

    # After rerun, reassign df from session state
    df = st.session_state.cleaned_df if st.session_state.cleaned and st.session_state.cleaned_df is not None else st.session_state.df

    if st.checkbox("Drop columns?"):
        cols = st.multiselect("Select columns to drop", options=df.columns.tolist())
        if st.button("Drop Selected Columns"):
            if cols:
                # Drop from main df
                st.session_state.df.drop(columns=cols, inplace=True)

                # If cleaned_df exists, drop from it too
                if st.session_state.cleaned_df is not None:
                    missing_in_cleaned = [c for c in cols if c in st.session_state.cleaned_df.columns]
                    if missing_in_cleaned:
                        st.session_state.cleaned_df.drop(columns=missing_in_cleaned, inplace=True)

                st.success(f"Dropped columns: {cols}")
                st.rerun()
            else:
                st.warning("No columns selected.")

    if st.session_state.cleaned:
        if st.button("Show Clean Data"):
            st.dataframe(st.session_state.cleaned_df.head(20))
