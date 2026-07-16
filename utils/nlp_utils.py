"""
utils/nlp_utils.py

Handles one-time NLTK resource downloads, stopword/lemmatizer setup,
and the `clean_NLP` text-cleaning function used by the NLP pages.
"""

import re

import nltk
import emoji
import contractions
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# ---- Downloads (only once) ----
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# ---- Stopwords setup ----
stop_words = set(stopwords.words('english'))
conflict_words = {
    'a', 'an', 'y', 'd', 've', 't', 's', 're', 'o', 'to', 'so', 'or', 'no',
    'm', 'ma', 'me', 'll', 'i', 'a', 'all', 'an', 'at', 'by'
}
stop_words.update(conflict_words)

lemmatizer = WordNetLemmatizer()


def clean_NLP(text):
    """Clean a single text string: lowercase, expand contractions, strip
    URLs/HTML, demojize, strip non-alpha chars, tokenize, remove stopwords
    and short tokens, and lemmatize. Returns "" for non-string input, and
    returns the (partially processed) text unchanged if an NLTK resource
    lookup fails.
    """
    if not isinstance(text, str):
        return ""
    try:
        # Lowercase
        text = text.lower()

        # Expand contractions
        text = contractions.fix(text)

        # Remove URLs and HTML
        text = re.sub(r"http\S+|www\S+|https\S+", '', text)
        text = re.sub(r'<.*?>', '', text)

        # Convert emojis to text labels
        text = emoji.demojize(text, delimiters=(" ", " "))

        # Remove digits and non-alphabetic (but keep underscores for emoji labels)
        text = re.sub(r'[^a-z_\s]', ' ', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Tokenize
        tokens = word_tokenize(text)

        # Remove stopwords + very short tokens
        tokens = [word for word in tokens if word not in stop_words and len(word) > 2]

        # Lemmatization
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

        return " ".join(tokens)
    except LookupError:
        return text
