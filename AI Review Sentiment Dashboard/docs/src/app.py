import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from transformers import pipeline
import time
import io

# Setup
st.set_page_config(page_title="Review Sentiment Dashboard", layout="wide")
analyzer = SentimentIntensityAnalyzer()
bert_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
stopwords = set(STOPWORDS)

# Plan selection
st.sidebar.header("Select Pricing Plan")
plan = st.sidebar.selectbox("Choose Plan", ["Free (100 reviews)", "Basic (500 reviews)", "Premium (1000 reviews)"])
plan_limits = {"Free (100 reviews)": 100, "Basic (500 reviews)": 500, "Premium (1000 reviews)": 1000}
MAX_REVIEWS = plan_limits[plan]

st.title("🧠 AI Review Sentiment Dashboard")
st.write("Upload a review CSV to analyze sentiments, word clouds, and download a report.")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        columns = df.columns.tolist()
        review_col = st.selectbox("Select Review Text Column", options=columns)
        date_col = st.selectbox("Select Date Column (optional)", options=["None"] + columns)
        company_col = st.selectbox("Select Company Column (optional)", options=["None"] + columns)

        # Clean data
        df = df.dropna(subset=[review_col])
        df[review_col] = df[review_col].astype(str)
        if date_col != "None":
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df['ReviewDate'] = df[date_col]
        else:
            df['ReviewDate'] = pd.date_range(start="2025-01-01", periods=len(df))
        df['Company'] = df[company_col] if company_col != "None" else "All"

        # Apply review limit
        if len(df) > MAX_REVIEWS:
            st.warning(f"Your {plan} allows only {MAX_REVIEWS} reviews. Processing the first {MAX_REVIEWS} reviews.")
            df = df.head(MAX_REVIEWS)
            time.sleep(2)

        # Sentiment analysis
        @st.cache_data
        def analyze_sentiments(reviews):
            def analyze_sentiment(text):
                vader_score = analyzer.polarity_scores(text)
                vader_sentiment = 'Positive' if vader_score['compound'] >= 0.05 else 'Negative' if vader_score['compound'] <= -0.05 else 'Neutral'
                blob = TextBlob(text)
                blob_sentiment = 'Positive' if blob.sentiment.polarity > 0 else 'Negative' if blob.sentiment.polarity < 0 else 'Neutral'
                bert_result = bert_analyzer(text[:512])[0]
                bert_sentiment = 'Positive' if bert_result['label'] == 'POSITIVE' else 'Negative'
                return {
                    'VADER': vader_sentiment, 'TextBlob': blob_sentiment, 'BERT': bert_sentiment,
                    'VADER_Score': vader_score['compound'], 'TextBlob_Score': blob.sentiment.polarity, 'BERT_Score': bert_result['score']
                }
            return [analyze_sentiment(text) for text in reviews]

        with st.spinner("Analyzing sentiments..."):
            df['Sentiment'] = analyze_sentiments(df[review_col].tolist())
            df['Sentiment_VADER'] = df['Sentiment'].apply(lambda x: x['VADER'])
            df['Sentiment_TextBlob'] = df['Sentiment'].apply(lambda x: x['TextBlob'])
            df['Sentiment_BERT'] = df['Sentiment'].apply(lambda x: x['BERT'])
            df['Month'] = df['ReviewDate'].dt.to_period('M').astype(str)

        # Company filter
        company_filter = st.selectbox("Select Company", ["All"] + df['Company'].unique().tolist())
        filtered_df = df if company_filter == "All" else df[df['Company'] == company_filter]

        # Preview
        st.subheader("📝 Review List with Sentiment")
        st.dataframe(filtered_df[[review_col, 'Sentiment_VADER', 'Sentiment_TextBlob', 'Sentiment_BERT', 'ReviewDate', 'Company']].head(20))

        # Sentiment counts
        st.subheader("📊 Sentiment Count Summary")
        st.write("VADER Sentiment Counts:")
        st.write(filtered_df['Sentiment_VADER'].value_counts())
        st.write("TextBlob Sentiment Counts:")
        st.write(filtered_df['Sentiment_TextBlob'].value_counts())
        st.write("BERT Sentiment Counts:")
        st.write(filtered_df['Sentiment_BERT'].value_counts())

        # Monthly breakdown
        st.subheader("📆 Monthly Sentiment Breakdown")
        month_summary = filtered_df.groupby(['Month', 'Sentiment_VADER']).size().unstack().fillna(0)
        st.dataframe(month_summary)

        # Trend chart
        st.subheader("📈 Sentiment Trend Over Time")
        trend_data = filtered_df.groupby([pd.Grouper(key='ReviewDate', freq='D'), 'Sentiment_VADER']).size().unstack().fillna(0)
        st.line_chart(trend_data)

        # Word cloud
        st.subheader("☁️ Word Cloud")
        sentiment_filter = st.selectbox("Filter Word Cloud by Sentiment", ["All", "Positive", "Negative", "Neutral"])
        wc_df = filtered_df
        if sentiment_filter != "All":
            wc_df = wc_df[wc_df['Sentiment_VADER'] == sentiment_filter]
        all_text = ' '.join(wc_df[review_col])
        if all_text.strip():
            wc = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords).generate(all_text)
            fig, ax = plt.subplots()
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.warning("No text available for the selected filters.")

        # Download report
        st.subheader("⬇️ Download Report")
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            filtered_df.to_excel(writer, index=False, sheet_name='ReviewData')
            filtered_df['Sentiment_VADER'].value_counts().to_frame(name='Count').to_excel(writer, sheet_name='SentimentCounts')
            month_summary.to_excel(writer, sheet_name='MonthlyBreakdown')
        st.download_button(
            label="Download Excel Report",
            data=output.getvalue(),
            file_name="sentiment_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error(f"Error processing CSV: {str(e)}")