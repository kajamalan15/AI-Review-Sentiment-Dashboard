import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from transformers import pipeline
import time
import io
import uuid

# Setup
st.set_page_config(page_title="Sentiment Insights", layout="wide")
analyzer = SentimentIntensityAnalyzer()
bert_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
stopwords = set(STOPWORDS)

# Theme management
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

# Define CSS for light and dark modes
light_mode_css = """
<style>
    .main {
        background-color: #ffffff;
        color: #000000;
    }
    .stSidebar {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .stSelectbox, .stFileUploader {
        background-color: #ffffff;
        color: #000000;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #000000;
    }
    .download-container {
        position: fixed;
        top: 10px;
        right: 10px;
        z-index: 1000;
    }
    .download-container .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
    }
</style>
"""

dark_mode_css = """
<style>
    .main {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    .stSidebar {
        background-color: #2c2c2c;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .stSelectbox, .stFileUploader {
        background-color: #333333;
        color: #ffffff;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff;
    }
    .download-container {
        position: fixed;
        top: 10px;
        right: 10px;
        z-index: 1000;
    }
    .download-container .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
    }
</style>
"""

# Apply theme based on session state
def apply_theme():
    if st.session_state.theme == 'dark':
        st.markdown(dark_mode_css, unsafe_allow_html=True)
    else:
        st.markdown(light_mode_css, unsafe_allow_html=True)

# Theme toggle button in sidebar
st.sidebar.header("Theme Settings")
theme_button = st.sidebar.button("Toggle Dark/Light Mode")
if theme_button:
    st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'
apply_theme()

# Plan selection
st.sidebar.header("Select Pricing Plan")
plan = st.sidebar.selectbox("Choose Plan", ["Free (100 reviews)", "Basic (500 reviews)", "Premium (1000 reviews)"])
plan_limits = {"Free (100 reviews)": 100, "Basic (500 reviews)": 500, "Premium (1000 reviews)": 1000}
MAX_REVIEWS = plan_limits[plan]

# Navigation
PAGES = {
    "Home": "Analyze Customer Sentiment Instantly",
    "Dashboard": "Your Review Sentiment Dashboard"
}
if 'page' not in st.session_state:
    st.session_state.page = "Home"
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

page = st.sidebar.selectbox("Navigate", list(PAGES.keys()), key="page_select", on_change=lambda: setattr(st.session_state, 'page', st.session_state.page_select))

# Page 1: Home (Analyze Customer Sentiment Instantly)
if page == "Home":
    st.title("Analyze Customer Sentiment Instantly")
    st.write("Upload your customer reviews and get powerful sentiment analysis insights powered by AI.")
    
    # File upload section
    st.markdown("""
    <div style='border: 2px dashed #ccc; padding: 20px; text-align: center; margin: 20px 0;'>
        <p>Drag & drop your review file here</p>
        <p>CSV or TXT supported, up to 10MB.</p>
    </div>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Browse Files", type=["csv", "txt"])
    
    if uploaded_file:
        st.session_state.uploaded_file = uploaded_file
        st.session_state.page = "Dashboard"
        st.rerun()
    
    # How It Works section
    st.subheader("How It Works")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**Upload Your Data**")
        st.write("Easily drag and drop or select your files in CSV or TXT format.")
    with col2:
        st.markdown("**Fast Analysis**")
        st.write("Our AI engine quickly processes your reviews to identify sentiment.")
    with col3:
        st.markdown("**Visualize Insights**")
        st.write("Get clear charts and summaries of sentiment trends and key topics.")
    with col4:
        st.markdown("**Detailed Review Table**")
        st.write("Explore individual reviews with their predicted sentiment labels in a searchable table.")

# Page 2: Dashboard
elif page == "Dashboard":
    st.title("🧠 AI Review Sentiment Dashboard")
    st.write("Upload a review CSV to analyze sentiments, word clouds, and download a report.")

    # File upload (using the file from Home page if available)
    uploaded_file = st.session_state.uploaded_file if st.session_state.uploaded_file else st.file_uploader("Upload your CSV file", type=["csv"])

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

            # Download report (positioned in top-right corner)
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                filtered_df.to_excel(writer, index=False, sheet_name='ReviewData')
                filtered_df['Sentiment_VADER'].value_counts().to_frame(name='Count').to_excel(writer, sheet_name='SentimentCounts')
                month_summary = filtered_df.groupby(['Month', 'Sentiment_VADER']).size().unstack().fillna(0)
                month_summary.to_excel(writer, sheet_name='MonthlyBreakdown')
            st.markdown('<div class="download-container">', unsafe_allow_html=True)
            st.download_button(
                label="Download Excel Report",
                data=output.getvalue(),
                file_name="sentiment_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            st.markdown('</div>', unsafe_allow_html=True)

            # Review preview
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

            # Pie Chart
            st.subheader("🥧 VADER Sentiment Pie Chart")
            vader_counts = filtered_df['Sentiment_VADER'].value_counts().reset_index()
            vader_counts.columns = ['Sentiment', 'Count']
            fig = px.pie(vader_counts, names='Sentiment', values='Count', title='Sentiment Distribution (VADER)')
            st.plotly_chart(fig)

            # Bar Chart: Sentiment by Company
            st.subheader("🏢 Sentiment by Company (VADER)")
            company_sentiment = filtered_df.groupby(['Company', 'Sentiment_VADER']).size().reset_index(name='Count')
            fig = px.bar(company_sentiment, x='Company', y='Count', color='Sentiment_VADER', barmode='group')
            st.plotly_chart(fig)

            # Monthly breakdown
            st.subheader("📆 Monthly Sentiment Breakdown")
            month_summary = filtered_df.groupby(['Month', 'Sentiment_VADER']).size().unstack().fillna(0)
            st.dataframe(month_summary)

            # Monthly Stacked Bar Chart
            st.subheader("📊 Monthly Sentiment Bar Chart")
            fig = px.bar(month_summary, x=month_summary.index, y=month_summary.columns,
                        title="Monthly VADER Sentiment", labels={'value': 'Count', 'Month': 'Month'})
            st.plotly_chart(fig)

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
                wc = WordCloud(width=800, height=400, background_color='white' if st.session_state.theme == 'light' else 'black', stopwords=stopwords).generate(all_text)
                fig, ax = plt.subplots()
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            else:
                st.warning("No text available for the selected filters.")

            # Interactive Explorer
            st.subheader("🔍 Explore Reviews by Sentiment and Keyword")
            selected_sentiment = st.selectbox("Filter by VADER Sentiment", ["All", "Positive", "Negative", "Neutral"])
            keyword = st.text_input("Search keyword in reviews")

            explore_df = filtered_df
            if selected_sentiment != "All":
                explore_df = explore_df[explore_df['Sentiment_VADER'] == selected_sentiment]
            if keyword:
                explore_df = explore_df[explore_df[review_col].str.contains(keyword, case=False)]

            st.write(f"Found {len(explore_df)} matching reviews:")
            st.dataframe(explore_df[[review_col, 'Sentiment_VADER', 'ReviewDate', 'Company']].head(50))

        except Exception as e:
            st.error(f"Error processing CSV: {str(e)}")