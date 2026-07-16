"""
pages/2_Dashboard.py

Tabular "Dashboard" page: main boxplot, per-column categorical counts,
numeric scatter/box/hist plots vs target, categorical top-category
summary, correlation heatmap, pairplot, and feature-importance chart.
Extracted from final_project.py.

Relies on st.session_state.df / target_col having been set on the
main app.py page (file upload + target selection happen there), and
on st.session_state.cleaned / cleaned_df if the user has already run
the cleaning step on the Tabular EDA page.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

from utils.state_utils import init_session_state
from utils.data_utils import get_sample
from utils.config import MAX_SAMPLE, PALETTE

st.set_page_config(page_title="Auto-Machine learning app", layout="wide")
sns.set_style("whitegrid")

init_session_state()

st.title("📊 Dashboard")

if st.session_state.df is None:
    st.info("Upload and process a dataset on the 'Tabular EDA' page first.")
else:
    if st.session_state.cleaned and st.session_state.cleaned_df is not None:
        df = st.session_state.cleaned_df.copy()
    else:
        df = st.session_state.df.copy()

    target_col = st.session_state.target_col

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Main Boxplot
    st.subheader("Main Boxplot (Categorical X / Numeric Y)")
    with st.expander("Customize main boxplot", expanded=True):
        main_cat = st.selectbox(
            "Choose categorical column (X)",
            options=categorical_cols,
            index=0 if categorical_cols else None,
            key="main_cat"
        )
        main_num = st.selectbox(
            "Choose numeric column (Y)",
            options=numeric_cols,
            index=0 if numeric_cols else None,
            key="main_num"
        )

    if main_cat and main_num:
        sample_df = get_sample(df[[main_cat, main_num]]) if df.shape[0] > MAX_SAMPLE else df[[main_cat, main_num]]
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=sample_df, x=main_cat, y=main_num, palette=PALETTE, ax=ax)
        ax.set_title(f"{main_num} by {main_cat}", fontsize=12)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        if df.shape[0] > MAX_SAMPLE:
            st.info(f"Main boxplot sampled to {MAX_SAMPLE} rows for speed.")

    st.markdown("---")
    # All categorical count plots
    st.subheader("All Categorical Columns — Counts")
    if categorical_cols:
        cols = st.columns(4)
        sample_for_cat = get_sample(df[categorical_cols]) if df.shape[0] > MAX_SAMPLE else df[categorical_cols]
        for i, cat in enumerate(categorical_cols):
            col = cols[i % 4]
            with col:
                fig, ax = plt.subplots(figsize=(3, 2))
                order = sample_for_cat[cat].value_counts().index[:20]
                sns.countplot(data=sample_for_cat, y=cat, order=order, palette=PALETTE, ax=ax)
                ax.set_title(cat, fontsize=9)
                ax.set_xlabel("")
                ax.set_ylabel("")
                plt.tight_layout()
                st.pyplot(fig)
    else:
        st.info("No categorical columns available.")

    st.markdown("---")
    # Numeric plots
    st.subheader("Numeric columns: scatter vs target or histograms")
    if numeric_cols:
        if target_col is not None and target_col in numeric_cols:
            cols_to_plot = [c for c in numeric_cols if c != target_col]
            sample_num = get_sample(df[cols_to_plot + [target_col]]) \
                if df.shape[0] > MAX_SAMPLE else df[cols_to_plot + [target_col]]

            cols_grid = st.columns(3)
            for i, num in enumerate(cols_to_plot):
                with cols_grid[i % 3]:
                    fig, ax = plt.subplots(figsize=(4, 3))
                    sns.scatterplot(data=sample_num, x=num, y=target_col, alpha=0.6)
                    ax.set_title(f"{num} vs {target_col}", fontsize=9)
                    st.pyplot(fig)

            if df.shape[0] > MAX_SAMPLE:
                st.info(f"Scatter plots sampled to {MAX_SAMPLE} rows for speed.")

        elif target_col is not None and target_col in categorical_cols:
            cols_grid = st.columns(3)
            sample_num = get_sample(df[numeric_cols + [target_col]]) if df.shape[0] > MAX_SAMPLE else df[numeric_cols + [target_col]]
            for i, num in enumerate(numeric_cols):
                with cols_grid[i % 3]:
                    fig, ax = plt.subplots(figsize=(4, 3))
                    sns.boxplot(data=sample_num, x=target_col, y=num, palette=PALETTE, ax=ax)
                    ax.set_title(f"{num} by {target_col}", fontsize=9)
                    st.pyplot(fig)
            if df.shape[0] > MAX_SAMPLE:
                st.info(f"Boxplots sampled to {MAX_SAMPLE} rows for speed.")
        else:
            pair_sample = get_sample(df[numeric_cols]) if df.shape[0] > MAX_SAMPLE else df[numeric_cols]
            show_cols = numeric_cols[:6]
            cols_grid = st.columns(3)
            for i, num in enumerate(show_cols):
                with cols_grid[i % 3]:
                    fig, ax = plt.subplots(figsize=(4, 3))
                    sns.histplot(pair_sample[num].dropna(), kde=True, ax=ax, stat="density")
                    ax.set_title(num, fontsize=9)
                    st.pyplot(fig)
            if df.shape[0] > MAX_SAMPLE:
                st.info(f"Numeric summaries sampled to {MAX_SAMPLE} rows for speed.")
    else:
        st.info("No numeric columns available.")

    st.markdown("---")
    # Categorical summary
    st.subheader("Categorical columns — top category counts summary")
    if categorical_cols:
        top_counts = {}
        for c in categorical_cols:
            top = df[c].value_counts().nlargest(1)
            top_counts[c] = top.index[0] if not top.empty else ""
        summary_df = pd.DataFrame({"Column": list(top_counts.keys()), "Top category": list(top_counts.values())})
        st.table(summary_df)
    else:
        st.info("No categorical columns.")

    st.markdown("---")
    # Correlation heatmap
    st.subheader("Correlation Heatmap (numeric)")
    if len(numeric_cols) > 1:
        sample_corr = get_sample(df[numeric_cols]) if df.shape[0] > MAX_SAMPLE else df[numeric_cols]
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(sample_corr.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.info("Need at least 2 numeric columns for correlation.")

    st.markdown("---")
    # Pairplot
    st.subheader("Pairplot (sampled)")
    if len(numeric_cols) > 1:
        pair_df = get_sample(df[numeric_cols]) if df.shape[0] > MAX_SAMPLE else df[numeric_cols]
        with st.spinner("Rendering pairplot (may take a moment)..."):
            g = sns.pairplot(pair_df.dropna(), plot_kws={"s": 10, "alpha": 0.6})
            st.pyplot(g.fig)
        if df.shape[0] > MAX_SAMPLE:
            st.info(f"Pairplot sampled to {MAX_SAMPLE} rows for speed.")
    else:
        st.info("Need at least 2 numeric columns for pairplot.")

    st.markdown("---")
    # Feature importance
    st.subheader("Feature importance (optional)")
    if target_col is None:
        st.info("Choose a target on the left sidebar to compute feature importance.")
    else:
        try:
            if target_col in df.columns:
                X = df.drop(columns=[target_col]).select_dtypes(include=[np.number, "object", "category"]).copy()
                X = pd.get_dummies(X, drop_first=True)
                y = df[target_col]
                if np.issubdtype(y.dtype, np.number) and y.nunique() > 10:
                    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                    task = "regression"
                else:
                    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                    task = "classification"
                if X.shape[0] > 2000:
                    Xs, _, ys, _ = train_test_split(
                        X, y, train_size=2000, random_state=42,
                        stratify=(y if task == "classification" and y.nunique() > 1 else None)
                    )
                else:
                    Xs, ys = X, y
                with st.spinner("Training model to compute feature importance..."):
                    model.fit(Xs, ys)
                importances = pd.Series(model.feature_importances_, index=Xs.columns).sort_values(ascending=False).head(30)
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.barplot(x=importances.values, y=importances.index, palette=PALETTE, ax=ax)
                ax.set_title("Top feature importances")
                st.pyplot(fig)
            else:
                st.warning("Selected target not present in current dataframe.")
        except Exception as e:
            st.error(f"Feature importance failed: {e}")

    # Footer: current shape
    active_df = st.session_state.cleaned_df if (st.session_state.cleaned and st.session_state.cleaned_df is not None) else st.session_state.df
    st.caption(f"Current dataset shape: {active_df.shape[0]} rows × {active_df.shape[1]} columns")
