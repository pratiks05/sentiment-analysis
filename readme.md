# Sentiment Analysis System

An end-to-end sentiment analysis application that supports both fast, rule-based sentiment detection and accurate transformer-based sentiment analysis. The system is designed to scale from single reviews to large CSV datasets and provides aspect-level insights.

---

## Features

- Dual sentiment engines:
  - Fast mode using VADER (NLTK)
  - Accurate mode using RoBERTa (Hugging Face)
- Positive, Neutral, Negative sentiment classification
- Aspect-based sentiment detection:
  - Acting
  - Story
  - Music
  - Direction
- Batch processing for large CSV files
- Confidence thresholding to flag uncertain predictions
- Interactive web interface built with Streamlit

---

## Tech Stack

- Python
- Streamlit
- Pandas
- Hugging Face Transformers
- NLTK (VADER)
- PyTorch

---

## Models Used

- VADER Sentiment Analyzer (NLTK)  
  - Rule-based, extremely fast, suitable for large datasets

- CardiffNLP RoBERTa Sentiment Model  
  - `cardiffnlp/twitter-roberta-base-sentiment`
  - Transformer-based, supports Negative / Neutral / Positive

---

## Application Workflow

1. User selects sentiment engine (Fast or Accurate)
2. User inputs text or uploads CSV
3. Text is validated and preprocessed
4. Sentiment is computed using selected model
5. Aspect-based analysis is applied
6. Results are visualized and downloadable

---

## Running Locally

```bash
python -m venv venv
source venv/bin/activate   # Windows: .\venv\Scripts\Activate
pip install -r requirements.txt
streamlit run app.py
