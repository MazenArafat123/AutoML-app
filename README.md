# 🤖 Auto-Machine Learning App

A no-code / low-code Streamlit application for exploring, cleaning, preprocessing, and
training machine learning models on **tabular** data (regression/classification) or
**text** data (NLP), powered by [PyCaret](https://pycaret.org/).

Upload a dataset, pick a target column, and let the app guide you through EDA,
cleaning, preprocessing, and automated model comparison — all from the browser,
no code required.

---

## ✨ Features

**Tabular pipeline**
- Data preview, `.info()` summary, missing-value detection, and descriptive statistics
- One-click automated cleaning (missing-value imputation, duplicate removal, outlier detection via log/IQR/z-score)
- Interactive dashboard: boxplots, categorical counts, scatter/histogram plots, correlation heatmap, pairplot, and Random Forest feature importance
- Column normalization (StandardScaler / PowerTransformer) and categorical encoding (One-Hot, Label, Mapping)
- Automated model comparison via PyCaret, with a leaderboard and a downloadable best model

**NLP pipeline**
- Data preview, missing-value handling, and descriptive statistics
- WordCloud visualization and target-distribution plotting
- Text cleaning (lowercasing, contraction expansion, URL/HTML/emoji handling, stopword removal, lemmatization)
- Automated model comparison via PyCaret on the selected text column, with a leaderboard and a downloadable best model

---

## 📁 Project Structure

```
.
├── app.py                          # Main entry point (Home page)
├── requirements.txt                # Python dependencies
├── .gitignore                      # Files/folders excluded from git
├── utils/                          # Shared helper modules (not Streamlit pages)
│   ├── config.py                   # Shared constants (MAX_SAMPLE, PALETTE)
│   ├── state_utils.py              # st.session_state initialization
│   ├── data_utils.py               # Tabular helpers: load, sample, clean, normalize
│   └── nlp_utils.py                # NLTK setup + clean_NLP text-cleaning function
└── pages/                          # Streamlit multipage app pages (auto-listed in sidebar)
    ├── 1_Tabular_EDA.py             # Tabular: preview, info, missing values, cleaning
    ├── 2_Dashboard.py               # Tabular: plots, heatmaps, feature importance
    ├── 3_Model_Preprocessing.py     # Tabular: scaling, encoding, CSV download
    ├── 4_Tabular_AutoML.py          # Tabular: PyCaret training, leaderboard, model download
    ├── 5_NLP_Overview.py            # NLP: preview, WordCloud, target distribution, cleaning
    └── 6_NLP_AutoML.py              # NLP: PyCaret training, leaderboard, model download
```

### What each file does

| File | Purpose |
|---|---|
| `app.py` | Sets page config, initializes session state, and hosts the sidebar **file uploader** and **target-column selector** shared by every page. This is the page Streamlit shows first (the "Home" page). |
| `utils/config.py` | Single source of truth for shared constants (`MAX_SAMPLE`, sample size cap for plotting; `PALETTE`, the seaborn color palette) so they aren't redefined in multiple files. |
| `utils/state_utils.py` | `init_session_state()` — ensures every `st.session_state` key the app relies on (`df`, `cleaned_df`, `target_col`, `encoded_df`, etc.) exists before it's read. |
| `utils/data_utils.py` | Tabular data helpers: `load_file()` (CSV/Excel loader), `get_sample()` (row sampler for fast plotting), `clean_and_analyze_normal()` (automated cleaning/outlier removal), `normalize_columns_inplace()` (scaling). |
| `utils/nlp_utils.py` | One-time NLTK resource downloads, stopword/lemmatizer setup, and the `clean_NLP()` text-cleaning function. |
| `pages/1_Tabular_EDA.py` | Data preview, `.info()`, missing-value table, descriptive stats, the "Clean & Analyze Data" button, and column dropping. |
| `pages/2_Dashboard.py` | Boxplots, categorical count plots, numeric scatter/box/histogram plots, correlation heatmap, pairplot, and feature-importance chart. |
| `pages/3_Model_Preprocessing.py` | Numeric column normalization, categorical encoding (One-Hot/Label/Mapping), and a button to download the preprocessed CSV. |
| `pages/4_Tabular_AutoML.py` | Detects regression vs. classification, runs PyCaret's `compare_models()`, displays the leaderboard, and offers the best model as a downloadable `.pkl`. |
| `pages/5_NLP_Overview.py` | Data preview, missing values, WordCloud, target distribution, and the text-cleaning button (uses `clean_NLP`). |
| `pages/6_NLP_AutoML.py` | Text-column selection, task detection, PyCaret training on the selected text + target columns, leaderboard, and best-model download. |
| `requirements.txt` | Pinned/scoped Python package list needed to run the app. |
| `.gitignore` | Excludes virtual environments, cache files, local datasets, and generated model artifacts from version control. |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv

# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** PyCaret is pinned to `3.4.0` (the classic functional-API line). Don't
> upgrade it independently — a newer major version may use an incompatible API.

### 4. Run the app
```bash
streamlit run app.py
```

Streamlit will open the app in your browser (default: `http://localhost:8501`).

### 5. Use the app
1. On the **Home** page, upload a CSV or Excel file and select a target column in the sidebar.
2. Use the sidebar page list to navigate:
   - **Tabular flow:** `Tabular EDA` → `Dashboard` → `Model Preprocessing` → `Tabular AutoML`
   - **NLP flow:** `NLP Overview` → `NLP AutoML`
3. Download the preprocessed dataset and/or the best trained model directly from the relevant page.

---

## 🛠️ Tech Stack

- **App framework:** Streamlit
- **Data handling:** pandas, NumPy, SciPy
- **Visualization:** Matplotlib, Seaborn, WordCloud
- **Machine learning / AutoML:** scikit-learn, PyCaret
- **NLP:** NLTK, emoji, contractions

---

## 📄 License

Add your preferred license here (e.g. MIT, Apache 2.0).
