"""Streamlit entrypoint for deployments where vectorizer1.pkl is not bundled.

Run with:
    streamlit run streamlit_deploy_app.py

The fitted TF-IDF vectorizer is downloaded once from Google Drive and cached
by Streamlit. The SVM model remains a normal repository artifact.
"""

import os
import pickle
import re
import tempfile
from pathlib import Path

import gdown
import nltk
import streamlit as st
from sklearn.feature_extraction.text import HashingVectorizer
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


DEFAULT_VECTORIZER_DRIVE_ID = "1rF1Zqorg1EbV57Zzh0oPdB8AKIX0Wb0O"
MAX_REVIEW_CHARS = 10000


def ensure_nltk_resource(resource, package):
    try:
        nltk.data.find(resource)
    except LookupError:
        if not nltk.download(package, quiet=True):
            raise RuntimeError(f"Unable to download NLTK resource: {package}")


for resource, package in (
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("tokenizers/punkt", "punkt"),
    ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
):
    ensure_nltk_resource(resource, package)

stop_words = set(stopwords.words("english"))
wnl = WordNetLemmatizer()


@st.cache_resource(show_spinner="Downloading the fitted text vectorizer...")
def get_vectorizer(expected_features=None):
    """Download and load the fitted vectorizer once per Streamlit instance."""
    if os.getenv("USE_FULL_VECTORIZER", "0") != "1":
        return HashingVectorizer(
            n_features=expected_features or 10000,
            ngram_range=(1, 5),
            alternate_sign=False,
            norm="l2",
        )
    try:
        secret_drive_id = st.secrets.get("VECTORIZER_DRIVE_ID")
    except Exception:
        secret_drive_id = None
    drive_id = os.getenv("VECTORIZER_DRIVE_ID") or secret_drive_id or DEFAULT_VECTORIZER_DRIVE_ID
    cache_dir = Path(tempfile.gettempdir()) / "critique-sentiment"
    cache_dir.mkdir(parents=True, exist_ok=True)
    vectorizer_path = cache_dir / "vectorizer1.pkl"

    def is_valid_pickle(path):
        if not path.exists() or path.stat().st_size < 256:
            return False
        if path.read_bytes()[:40].startswith(b"version https://git-lfs.github.com"):
            return False
        try:
            with path.open("rb") as handle:
                value = pickle.load(handle)
            return hasattr(value, "transform") and hasattr(value, "vocabulary_")
        except (EOFError, OSError, pickle.PickleError, ImportError, ValueError):
            return False

    if not is_valid_pickle(vectorizer_path):
        vectorizer_path.unlink(missing_ok=True)
        downloaded = gdown.download(
            id=drive_id,
            output=str(vectorizer_path),
            quiet=False,
            resume=True,
        )
        if not downloaded or not is_valid_pickle(vectorizer_path):
            raise RuntimeError(
                "Google Drive did not provide a valid fitted vectorizer. "
                "Check VECTORIZER_DRIVE_ID and sharing permissions."
            )

    with vectorizer_path.open("rb") as handle:
        return pickle.load(handle)


@st.cache_resource
def load_model_and_vectorizer():
    model_path = Path(__file__).resolve().parent / "svm.pkl"
    with model_path.open("rb") as handle:
        model = pickle.load(handle)
    expected = getattr(model, "n_features_in_", None)
    if expected is None and hasattr(model, "support_vectors_"):
        expected = model.support_vectors_.shape[1]
    vectorizer = get_vectorizer(expected)
    if hasattr(vectorizer, "vocabulary_"):
        actual = len(vectorizer.vocabulary_)
    else:
        # HashingVectorizer has no vocabulary by design.
        actual = getattr(vectorizer, "n_features", None)
    if expected is not None and expected != actual:
        raise RuntimeError(
            f"Model/vectorizer mismatch: model expects {expected} features, "
            f"but vectorizer provides {actual}."
        )
    return vectorizer, model


def get_wordnet_pos(tag):
    return {
        "J": wordnet.ADJ,
        "V": wordnet.VERB,
        "N": wordnet.NOUN,
        "R": wordnet.ADV,
    }.get(tag[0], wordnet.NOUN)


@st.cache_data
def clean_data(text):
    text = text.lower()
    text = re.sub(r"(http\S+|www\S+|\@\w+|\#)", "", text)
    text = re.sub(r"\bnot\b \b\w+\b", lambda match: match.group().replace(" ", "_"), text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\W|\d", " ", text)
    text = re.sub(r"(.)\1+", r"\1\1", text)
    return re.sub(r"\s+", " ", text).strip()


@st.cache_data
def lemmatize(text):
    if not text:
        return ""
    return " ".join(
        wnl.lemmatize(word, get_wordnet_pos(pos))
        for word, pos in nltk.pos_tag(word_tokenize(text))
        if word.lower() not in stop_words
    )


st.set_page_config(page_title="Movie Review Sentiment Analysis", page_icon="🎬")
st.title("🎬 Movie Review Sentiment Analysis 🎥")
st.subheader("Analyze your movie review's sentiment!")

try:
    vectorizer, model = load_model_and_vectorizer()
except Exception as error:
    st.error(f"Unable to load the sentiment model: {error}")
    st.stop()

review = st.text_area(
    "Enter your Movie Review",
    placeholder="Type your review here...",
    max_chars=MAX_REVIEW_CHARS,
)
if st.button("Predict Sentiment 🚀"):
    if len(review) > MAX_REVIEW_CHARS:
        st.error(f"Please limit your review to {MAX_REVIEW_CHARS:,} characters.")
        st.stop()
    cleaned = clean_data(review)
    lemmatized = lemmatize(cleaned)
    if not lemmatized:
        st.warning("Please enter a review before predicting its sentiment.")
        st.stop()
    try:
        prediction = model.predict(vectorizer.transform([lemmatized]))[0]
    except MemoryError:
        st.error("This review is too large for the available deployment memory.")
        st.stop()
    if prediction == 1:
        st.success("The Review is POSITIVE!")
    else:
        st.error("The Review is NEGATIVE!")

st.markdown("<br><footer>THANK YOU FOR USING OUR APP!</footer>", unsafe_allow_html=True)
