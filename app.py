import streamlit as st
import pandas as pd
import math
import re

from transformers import pipeline
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# --------------------------------------------------
# INITIAL SETUP
# --------------------------------------------------
st.set_page_config(
    page_title="Sentiment Analysis System",
    layout="wide"
)

st.title("Sentiment Analysis System")
st.write("Scalable sentiment analysis with fast and accurate model selection.")

# --------------------------------------------------
# DOWNLOAD NLTK DATA (ONCE)
# --------------------------------------------------
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download("vader_lexicon", quiet=True)

# --------------------------------------------------
# LOAD MODELS (CACHED)
# --------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_vader():
    return SentimentIntensityAnalyzer()

@st.cache_resource(show_spinner=False)
def load_transformer():
    try:
        # Using a smaller, faster model that's more deployment-friendly
        model = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1  # Force CPU usage
        )
        return model
    except Exception as e:
        st.error(f"Error loading transformer model: {str(e)}")
        st.info("Falling back to VADER model")
        return None

LABEL_MAP = {
    "LABEL_0": "NEGATIVE",
    "LABEL_1": "NEUTRAL",
    "LABEL_2": "POSITIVE",
    "NEGATIVE": "NEGATIVE",
    "POSITIVE": "POSITIVE"
}

# --------------------------------------------------
# ASPECT KEYWORDS
# --------------------------------------------------
ASPECTS = {
    "Acting": ["acting", "actor", "performance", "cast"],
    "Story": ["story", "plot", "screenplay", "script"],
    "Music": ["music", "songs", "background score", "bgm"],
    "Direction": ["direction", "director", "cinematography"]
}

def detect_aspects(text):
    found = []
    text = text.lower()
    for aspect, keywords in ASPECTS.items():
        if any(k in text for k in keywords):
            found.append(aspect)
    return found if found else ["General"]

# --------------------------------------------------
# SENTIMENT FUNCTIONS
# --------------------------------------------------
def vader_sentiment(text, vader):
    score = vader.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "POSITIVE", abs(score)
    elif score <= -0.05:
        return "NEGATIVE", abs(score)
    else:
        return "NEUTRAL", abs(score)

def transformer_sentiment(text, model):
    if model is None:
        st.error("Transformer model not available. Please use VADER.")
        return "NEUTRAL", 0.5
    
    # Truncate text to avoid token limits
    text = text[:512]
    out = model(text)[0]
    label = LABEL_MAP.get(out["label"], out["label"])
    return label, out["score"]

# --------------------------------------------------
# MODEL SELECTION
# --------------------------------------------------
model_type = st.radio(
    "Select Sentiment Engine",
    ["Fast (VADER)", "Accurate (Transformer)"],
    horizontal=True
)

# Load selected model with error handling
with st.spinner(f"Loading {model_type} model..."):
    if model_type == "Fast (VADER)":
        sentiment_engine = load_vader()
    else:
        sentiment_engine = load_transformer()
        if sentiment_engine is None:
            st.warning("Transformer model failed to load. Switching to VADER.")
            model_type = "Fast (VADER)"
            sentiment_engine = load_vader()

CONFIDENCE_THRESHOLD = 0.60

# --------------------------------------------------
# TABS
# --------------------------------------------------
tab1, tab2 = st.tabs(["Single Review", "CSV Dataset"])

# ==================================================
# TAB 1: SINGLE REVIEW
# ==================================================
with tab1:
    st.subheader("Single Review Sentiment Analysis")

    user_input = st.text_area(
        "Enter review text",
        "The acting was excellent but the story felt weak.",
        height=120
    )

    if st.button("Analyze Review", use_container_width=True):
        if user_input.strip():
            with st.spinner("Analyzing sentiment..."):
                if model_type == "Fast (VADER)":
                    label, score = vader_sentiment(user_input, sentiment_engine)
                else:
                    label, score = transformer_sentiment(user_input, sentiment_engine)

            if score < CONFIDENCE_THRESHOLD:
                label = "UNCERTAIN"

            aspects = detect_aspects(user_input)

            if label == "NEGATIVE":
                st.error(f"Sentiment: {label}")
            elif label == "NEUTRAL":
                st.warning(f"Sentiment: {label}")
            elif label == "POSITIVE":
                st.success(f"Sentiment: {label}")
            else:
                st.info("Sentiment: UNCERTAIN")

            st.info(f"Confidence Score: {score:.2f}")
            st.write(f"Detected Aspects: {', '.join(aspects)}")
        else:
            st.warning("Please enter review text.")

# ==================================================
# TAB 2: CSV ANALYSIS
# ==================================================
with tab2:
    st.subheader("CSV Sentiment Analysis")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        st.write("Dataset Preview")
        st.dataframe(df.head())

        text_column = st.selectbox(
            "Select review text column",
            df.columns
        )

        if pd.api.types.is_numeric_dtype(df[text_column]):
            st.error("Selected column is numeric. Please select a text column.")
            st.stop()

        max_rows = st.slider(
            "Rows to analyze",
            min_value=50,
            max_value=min(len(df), 2000),
            value=min(len(df), 500),
            step=50
        )

        if st.button("Analyze Dataset", use_container_width=True):
            texts = df[text_column].astype(str).head(max_rows).tolist()

            sentiments = []
            confidences = []
            aspects_list = []

            batch_size = 16 if model_type == "Accurate (Transformer)" else 1
            total_batches = math.ceil(len(texts) / batch_size)

            progress = st.progress(0)
            status = st.empty()

            with st.spinner("Running analysis..."):
                for i in range(total_batches):
                    batch = texts[i * batch_size:(i + 1) * batch_size]

                    if model_type == "Fast (VADER)":
                        for text in batch:
                            label, score = vader_sentiment(text, sentiment_engine)
                            sentiments.append(label)
                            confidences.append(score)
                            aspects_list.append(", ".join(detect_aspects(text)))
                    else:
                        for text in batch:
                            label, score = transformer_sentiment(text, sentiment_engine)
                            
                            if score < CONFIDENCE_THRESHOLD:
                                label = "UNCERTAIN"

                            sentiments.append(label)
                            confidences.append(score)
                            aspects_list.append(", ".join(detect_aspects(text)))

                    progress.progress((i + 1) / total_batches)
                    status.text(f"Processed batch {i + 1} of {total_batches}")

            result_df = df.loc[:max_rows - 1, [text_column]].copy()
            result_df["Sentiment"] = sentiments
            result_df["Confidence"] = confidences
            result_df["Aspect"] = aspects_list

            st.success("Sentiment analysis completed.")

            st.write("Results Preview")
            st.dataframe(result_df.head(10))

            st.write("Sentiment Distribution")
            st.bar_chart(result_df["Sentiment"].value_counts())

            csv_data = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Results CSV",
                csv_data,
                "sentiment_results.csv",
                "text/csv",
                use_container_width=True
            )