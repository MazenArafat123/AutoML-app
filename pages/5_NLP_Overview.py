"""
pages/5_NLP_Overview.py

NLP "Data Overview" page: data preview, dataset info, missing values
(with a drop-rows option), descriptive stats, WordCloud + target
distribution visualization, text-column cleaning via clean_NLP, and
column dropping. Extracted from final_project.py.

Relies on st.session_state.df / target_col having been set on the
main app.py page (file upload + target selection happen there).
"""

from io import StringIO

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from wordcloud import WordCloud

from utils.state_utils import init_session_state
from utils.nlp_utils import clean_NLP

st.set_page_config(page_title="Auto-Machine learning app", layout="wide")

init_session_state()

st.title("NLP Data overview")

if st.session_state.df is None:
    st.info("Upload a CSV / Excel file from the left sidebar to begin.")
else:
    df = st.session_state.df
    target_col = st.session_state.target_col

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
        if st.button("Drop rows with missing values"):
            df.dropna(inplace=True)
            st.session_state.df = df  # keep changes
            st.success("All rows with missing values have been dropped.")

    st.subheader("📊 Descriptive Statistics")
    st.write(df.describe(include="all"))

    # === 5. Visualization (WordCloud + Target) ===
    st.markdown("---")
    st.subheader("🖼️ Visualization")

    st.session_state.text_cols = [
        c for c in df.columns if df[c].dtype == "object" and c != st.session_state.target_col and df[c].nunique() > 15
    ]
    text_cols = st.session_state.text_cols

    if text_cols:
        text_col = st.selectbox("Select text column for WordCloud", options=text_cols)
        text_data = " ".join(df[text_col].dropna().astype(str).tolist())
        if text_data.strip():
            wc = WordCloud(width=800, height=400, background_color="white").generate(text_data)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

    if target_col is not None and target_col in df.columns:
        st.subheader(f"📌 Target distribution: {target_col}")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x=df[target_col], palette="pastel", ax=ax)
        ax.set_title(f"Distribution of {target_col}")
        st.pyplot(fig)

    # === 6. Cleaned Text Preview ===
    st.subheader("🧹 Cleaning Text Columns")
    if st.button("Clean all text columns (except target)"):
        for col in text_cols:
            df[col] = df[col].astype(str).apply(clean_NLP)
        st.session_state.df = df
        st.success(f"Cleaned columns: {text_cols}")
        st.dataframe(df.head(10))

    # === 7. Final Processing ===
    st.markdown("---")
    st.header("⚙️ Final Processing")

    if st.checkbox("Drop columns?"):
        cols = st.multiselect("Select columns to drop", options=df.columns.tolist())
        if st.button("Drop Selected Columns"):
            if cols:
                st.session_state.df.drop(columns=cols, inplace=True)
                st.success(f"Dropped columns: {cols}")
                st.rerun()
            else:
                st.warning("No columns selected.")
