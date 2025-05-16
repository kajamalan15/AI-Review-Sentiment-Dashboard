import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time
import io

# Constants
MAX_REVIEWS = 1000

# Setup
st.set_page_config(page_title="Review Sentiment Dashboard", layout="wide")
analyzer = SentimentIntensityAnalyzer()

st.title(" AI Review Sentiment Dashboard")
st.write("Upload a review CSV file to analyze sentiments, word clouds, and download a report.")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    columns = df.columns.tolist()

    # Column selectors
    review_col = st.selectbox("Select Review Text Column", options=columns)
    date_col = st.selectbox("Select Date Column (optional)", options=["None"] + columns)
    company_col = st.selectbox("Select Company Column (optional)", options=["None"] + columns)

    # Clean data
    df = df.dropna(subset=[review_col])
    df[review_col] = df[review_col].astype(str)

    # Handle date
    if date_col != "None":
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df['ReviewDate'] = df[date_col]
    else:
        df['ReviewDate'] = pd.date_range(start="2025-01-01", periods=len(df))

    # Handle company
    if company_col != "None":
        df['Company'] = df[company_col]
    else:
        df['Company'] = "All"

    # Respect review limit
    if len(df) > MAX_REVIEWS:
        st.warning(f"Only the first {MAX_REVIEWS} reviews will be processed.")
        df = df.head(MAX_REVIEWS)
        time.sleep(2)

 
   
