"""
utils/data_utils.py

Helper functions for loading, sampling, cleaning/analyzing, and
normalizing tabular data. Extracted from final_project.py.
"""

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
from sklearn.preprocessing import StandardScaler, PowerTransformer

from utils.config import MAX_SAMPLE


@st.cache_data
def load_file(uploaded_file):
    """Load an uploaded CSV or Excel file into a DataFrame."""
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)
    else:
        return pd.read_excel(uploaded_file)


def get_sample(df: pd.DataFrame, n=MAX_SAMPLE, random_state=42):
    """Return a random sample of at most n rows from df (or df itself if smaller)."""
    if df is None:
        return df
    if df.shape[0] <= n:
        return df.copy()
    return df.sample(n=n, random_state=random_state).copy()


# ---- PROCESSING FUNCTIONS ----
def clean_and_analyze_normal():
    """Clean st.session_state.df in place: drop high-missing columns, impute
    or drop remaining missing values, drop duplicates, and detect/remove
    outliers using whichever of log/IQR/z-score transforms yields the
    lowest mean std. Writes results back to st.session_state.df and
    st.session_state.cleaned_df.
    """
    df = st.session_state.df
    if df is None:
        return
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    # Drop columns with >45% missing
    col_thresh = len(df) * 0.45
    dropped_cols = df.columns[df.isnull().sum() > col_thresh].tolist()
    if dropped_cols:
        df.drop(columns=dropped_cols, inplace=True)
        st.success(f"Dropped columns with >45% missing values: {dropped_cols}")

    # Missing values
    total_nulls = int(df.isnull().sum().sum())
    null_pct = total_nulls / max(1, len(df))
    if total_nulls == 0:
        st.success("No missing values found.")
    elif null_pct < 0.10:
        removed_rows = df[df.isnull().any(axis=1)].shape[0]
        df.dropna(inplace=True)
        st.success(f"Removed {removed_rows} rows with missing values (<10% total).")
    else:
        for col in df.columns:
            cnt = int(df[col].isnull().sum())
            if cnt > 0:
                if np.issubdtype(df[col].dtype, np.number):
                    med = df[col].median()
                    df[col].fillna(med, inplace=True)
                    st.success(f"Filled {cnt} missing numeric values in '{col}' with median ({med}).")
                else:
                    mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else ""
                    df[col].fillna(mode_val, inplace=True)
                    st.success(f"Filled {cnt} missing categorical values in '{col}' with mode ('{mode_val}').")

    # Remove duplicates
    dup = int(df.duplicated().sum())
    if dup > 0:
        df.drop_duplicates(inplace=True)
        st.success(f"Removed {dup} duplicate rows.")

    # Outlier handling
    numeric_now = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_now:
        try:
            z_scores = np.abs(stats.zscore(df[numeric_now], nan_policy='omit'))
            outliers_mask = (z_scores > 3).any(axis=1) if z_scores.ndim > 1 else (z_scores > 3)
            outlier_pct = float(np.mean(outliers_mask) * 100)
        except Exception:
            outlier_pct = 0.0

        remove_outliers = True
        if outlier_pct > 25:
            choice = st.radio(f"High outlier percentage detected ({outlier_pct:.2f}%). Remove them?",
                              ["Yes", "No"], index=0, key="outlier_choice")
            remove_outliers = (choice == "Yes")

        if remove_outliers:
            candidates = {}
            # Log transform
            try:
                df_log = df.copy()
                valid = True
                for c in numeric_now:
                    if (df_log[c] <= 0).any():
                        valid = False
                        break
                    df_log[c] = np.log(df_log[c])
                if valid:
                    candidates["log"] = df_log
            except Exception:
                pass

            # IQR
            df_iqr = df.copy()
            for c in numeric_now:
                Q1 = df_iqr[c].quantile(0.25)
                Q3 = df_iqr[c].quantile(0.75)
                IQR = Q3 - Q1
                df_iqr = df_iqr[~((df_iqr[c] < (Q1 - 1.5 * IQR)) | (df_iqr[c] > (Q3 + 1.5 * IQR)))]
            candidates["iqr"] = df_iqr

            # Z-score
            try:
                df_z = df.copy()
                mask = np.ones(len(df_z), dtype=bool)
                for c in numeric_now:
                    col_z = np.abs(stats.zscore(df_z[c], nan_policy='omit'))
                    mask &= (col_z < 3)
                df_z = df_z[mask]
                candidates["zscore"] = df_z
            except Exception:
                pass

            # Choose best
            best_key, best_df, best_std = None, df, float("inf")
            for k, cand in candidates.items():
                num = cand.select_dtypes(include=np.number)
                if num.shape[0] == 0:
                    continue
                mean_std = float(num.std().mean())
                if mean_std < best_std:
                    best_std = mean_std
                    best_key = k
                    best_df = cand

            removed = df.shape[0] - best_df.shape[0]
            if removed > 0:
                st.session_state.df = best_df.reset_index(drop=True)
                st.session_state.cleaned_df = st.session_state.df.copy()
                st.success(f"Outlier removal applied using '{best_key}' method. Removed {removed} rows.")
                return

    st.session_state.df = df.reset_index(drop=True)
    st.session_state.cleaned_df = st.session_state.df.copy()


def normalize_columns_inplace(cols):
    """Normalize the given numeric columns of st.session_state.df in place,
    using PowerTransformer (yeo-johnson) for skewed columns (|skew| > 1,
    falling back to StandardScaler on failure) and StandardScaler otherwise.
    """
    df = st.session_state.df
    if df is None or not cols:
        return
    for c in cols:
        if c not in df.columns:
            continue
        if not np.issubdtype(df[c].dtype, np.number):
            st.warning(f"Skipped normalization for non-numeric column '{c}'.")
            continue
        skew = float(df[c].skew())
        if abs(skew) > 1:
            try:
                transformer = PowerTransformer(method='yeo-johnson')
                df[c] = transformer.fit_transform(df[[c]])
                st.success(f"Applied PowerTransformer to '{c}' (skew={skew:.2f}).")
            except Exception:
                scaler = StandardScaler()
                df[c] = scaler.fit_transform(df[[c]])
                st.success(f"PowerTransformer failed for '{c}'; used StandardScaler instead.")
        else:
            scaler = StandardScaler()
            df[c] = scaler.fit_transform(df[[c]])
            st.success(f"Standard scaled '{c}' (skew={skew:.2f}).")
    st.session_state.df = df
