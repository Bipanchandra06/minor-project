import streamlit as st
import pickle
import re
import os
from pathlib import Path
import nltk
from sklearn.feature_extraction.text import HashingVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet

MAX_REVIEW_CHARS = 10000

# Download NLTK resources only when missing. Newer NLTK releases split some
# resources into language-specific packages.
def ensure_nltk_resource(resource, package):
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(package, quiet=True)


ensure_nltk_resource('corpora/stopwords', 'stopwords')
ensure_nltk_resource('corpora/wordnet', 'wordnet')
ensure_nltk_resource('tokenizers/punkt', 'punkt')
ensure_nltk_resource('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger')
ensure_nltk_resource('tokenizers/punkt_tab', 'punkt_tab')
ensure_nltk_resource('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng')
stop_words = set(stopwords.words('english'))

# Initialize lemmatizer
wnl = WordNetLemmatizer()

# Set up Streamlit page
st.set_page_config(page_title="Movie Review Sentiment Analysis", page_icon="🎬")

# Load vectorizer and model
@st.cache_resource  # Cached only once for all users
def load_model_and_vectorizer():
    model_dir = Path(__file__).resolve().parent
    vectorizer_path = model_dir / 'vectorizer1.pkl'
    model_path = model_dir / 'svm.pkl'
    with model_path.open('rb') as model_file:
        model = pickle.load(model_file)
    feature_count = getattr(model, 'n_features_in_', None)
    if feature_count is None and hasattr(model, 'support_vectors_'):
        feature_count = model.support_vectors_.shape[1]

    if os.getenv('USE_FULL_VECTORIZER', '0') != '1':
        vectorizer = HashingVectorizer(
            n_features=feature_count or 10000,
            ngram_range=(1, 5),
            alternate_sign=False,
            norm='l2',
        )
        return vectorizer, model

    with vectorizer_path.open('rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    vectorizer_count = len(getattr(vectorizer, 'vocabulary_', {}))
    if feature_count is not None and vectorizer_count != feature_count:
        raise RuntimeError(
            f'Model/vectorizer mismatch: model expects {feature_count} features, '
            f'but vectorizer provides {vectorizer_count}.'
        )
    return vectorizer, model

vectorizer, model = load_model_and_vectorizer()

# Map POS tags for lemmatization
def get_wordnet_pos(tag):
    return {
        'J': wordnet.ADJ,
        'V': wordnet.VERB,
        'N': wordnet.NOUN,
        'R': wordnet.ADV
    }.get(tag[0], wordnet.NOUN)

# Clean data function
@st.cache_data
def clean_data(text):
    text = text.lower()
    text = re.sub(r'(http\S+|www\S+|\@\w+|\#)', '', text)  # Remove URLs, @, and hashtags
    text = re.sub(r'\bnot\b \b\w+\b', lambda x: x.group().replace(' ', '_'), text)  # "not" modifier
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags directly with regex
    text = re.sub(r'\W|\d', ' ', text)  # Remove special chars and digits
    text = re.sub(r'(.)\1+', r'\1\1', text)  # Remove repeated characters
    return re.sub(r'\s+', ' ', text).strip()

# Lemmatize text
@st.cache_data
def lemmatize(text):
    if not text:
        return ''
    tokens = [wnl.lemmatize(word, get_wordnet_pos(pos))
              for word, pos in nltk.pos_tag(word_tokenize(text))
              if word.lower() not in stop_words]
    return " ".join(tokens)

# Updated CSS styling for a new color scheme
st.markdown(
    """
 <style>
    /* Background styling */
    .main {
        background: linear-gradient(to bottom, #1e3c72, #2a5298);
        background-size: cover;
        padding: 20px;
    }
    /* Card and text styling */
    .stApp {
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 12px;
        padding: 2rem;
    }
    h1 {
        color: #ff6b6b;
        text-align: center;
        font-size: 50px;
        font-family: 'Courier New', Courier, monospace;
        font-weight: bold;
        text-shadow: 2px 2px 4px #000;
    }
    h2 {
        color: #f5f5f5;
        text-align: center;
        font-size: 30px;
        font-family: 'Arial', sans-serif;
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #ff6b6b;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-size: 18px;
        font-weight: bold;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.3);
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #ff8a8a;
    }
    /* Sentiment result styling */
    .stAlert {
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        animation: fadeIn 1s ease-in-out;
    }
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    footer {
        font-size: 18px;
        color: #ff6b6b;
        text-align: center;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True
)

st.title("🎬 Movie Review Sentiment Analysis 🎥")
st.subheader("Analyze your movie review's sentiment!")

# User input
review = st.text_area(
    "Enter your Movie Review",
    placeholder="Type your review here...",
    max_chars=MAX_REVIEW_CHARS,
)

# Predict sentiment
if st.button("Predict Sentiment 🚀"):
    with st.spinner('Analyzing your review...'):
        if len(review) > MAX_REVIEW_CHARS:
            st.error(f"Please limit your review to {MAX_REVIEW_CHARS:,} characters.")
            st.stop()

        # Clean and lemmatize the input text
        cleaned_data = clean_data(review)
        lemmatized_data = lemmatize(cleaned_data)

        if not lemmatized_data:
            st.warning("Please enter a review before predicting its sentiment.")
            st.stop()

        # Predict sentiment
        try:
            prediction = model.predict(vectorizer.transform([lemmatized_data]))[0]
        except MemoryError:
            st.error("This review is too large for the available deployment memory.")
            st.stop()

        # Set sentiment message based on prediction
        sentiment = " The Review is POSITIVE!" if prediction == 1 else " The Review is NEGATIVE!"

        # Display result based on sentiment
        if prediction == 1:
            st.success(sentiment)
        else:
            st.error(sentiment)

# Footer
st.markdown("<br><footer>THANK YOU FOR USING OUR APP!</footer>", unsafe_allow_html=True)
