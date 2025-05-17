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

# Define CSS for light and dark modes with improved alignments
light_mode_css = """
<style>
    .main {
        background-color: #f9f9f9;
        color: #333333;
        padding: 20px;
        min-height: 100vh;
    }
    .stSidebar {
        background-color: #e8ecef;
        padding: 15px;
        border-right: 1px solid #d1d3d4;
    }
    .stButton>button {
        background-color: #28a745;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #218838;
    }
    .stSelectbox, .stFileUploader {
        background-color: #ffffff;
        color: #333333;
        border: 1px solid #ced4da;
        border-radius: 5px;
        padding: 5px;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #2c3e50;
        margin-bottom: 15px;
    }
    .download-container {
        position: fixed;
        top: 15px;
        right: 20px;
        z-index: 1000;
        display: flex;
        justify-content: flex-end;
    }
    .download-container .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 500;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .download-container .stButton>button:hover {
        background-color: #0056b3;
    }
    .section-container {
        margin-bottom: 30px;
        padding: 20px;
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .how-it-works {
        display: flex;
        gap: 20px;
        flex-wrap: wrap;
        justify-content: space-between;
    }
    .how-it-works .stColumn {
        flex: 1;
        min-width: 200px;
        padding: 15px;
        background-color: #f1f3f5;
        border-radius: 8px;
    }
    @media (max-width: 768px) {
        .download-container {
            position: static;
            margin: 10px 0;
            justify-content: center;
        }
        .how-it-works {
            flex-direction: column;
        }
    }
</style>
"""

dark_mode_css = """
<style>
    .main {
        background-color: #222222;
        color: #e0e0e0;
        padding: 20px;
        min-height: 100vh;
    }
    .stSidebar {
        background-color: #2d2d2d;
        padding: 15px;
        border-right: 1px solid #444444;
    }
    .stButton>button {
        background-color: #28a745;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #218838;
    }
    .stSelectbox, .stFileUploader {
        background-color: #333333;
        color: #e0e0e0;
        border: 1px solid #555555;
        border-radius: 5px;
        padding: 5px;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff;
        margin-bottom: 15px;
    }
    .download-container {
        position: fixed;
        top: 15px;
        right: 20px;
        z-index: 1000;
        display: flex;
        del-right: 20px;
        z-index: 1000;
        display: flex;
        justify-content: flex-end;
    }
    .download-container .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 500;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    .download-container .stButton>button:hover {
        background-color: #0056b3;
    }
    .section-container {
        margin-bottom: 30px;
        padding: 20px;
        background-color: #2c2c2c;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    .how-it-works {
        display: flex;
        gap: 20px;
        flex-wrap: wrap;
        justify-content: space-between;
    }
    .how-it-works .stColumn {
        flex: 1;
        min-width: 200px;
        padding: 15px;
        background-color: #3a3a3a;
        border-radius: 8px;
    }
    @media (max-width: 768px) {
        .download-container {
            position: static;
            margin: 10px 0;
            justify-content: center;
        }
        .how-it-works {
            flex-direction: column;
        }
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
    <div style='border: 2px dashed #ccc; padding: 20px; text-align: center; margin: 20px 0; border-radius: 8px;'>
        <p>Drag & drop your review file here</p>
        <p>CSV or TXT supported, up to 10MB.</p>
    </div>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Browse Files", type=["csv", "txt"])
    
    if uploaded_file:
        st.session_state.uploaded_file = uploaded_file
        st.session_state.page = "Dashboard"
        st.rerun()
    
    # How It Works section with improved alignment
    st.subheader("How It Works")
    st.markdown('<div class="how-it-works">', unsafe_allow_html=True)
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
    st.markdown('</div>', unsafe_allow_html=True)

# Page 2: Dashboard
elif page == "Dashboard":
    st.title(" AI Review Sentiment Dashboard")
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
            st.markdown('<div class="section-container">', unsafe_allow_html=True)
            st.subheader(" Review List with Sentiment")
            st.dataframe(filtered_df[[review_col, 'Sentiment_VADER', 'Sentiment_TextBlob', 'Sentiment_BERT', 'ReviewDate', 'Company']].head(20))
            st.markdown('</div>', unsafe_allow_html=True)

            # Sentiment counts
            st.markdown('<div class="section-container">', unsafe_allow_html=True)
            st.subheader(" Sentiment Count Summary")
            st.write("VADER Sentiment Counts:")
            st.write(filtered_df['Sentiment_VADER'].value_counts())
            st.write("TextBlob Sentiment Counts:")
            st.write(filtered_df['Sentiment_TextBlob'].value_counts())
            st.write("BERT Sentiment Counts:")
            st.write(filtered_df['Sentiment_BERT'].value_counts())
            st.markdown('</div>', unsafe_allow_html=True)

            # Pie Chart
            st.markdown('<div class="section-container">', unsafe_allow_html=True)
            st.subheader(" VADER Sentiment Pie Chart")
            vader_counts = filtered_df['Sentiment_VADER'].value_counts().reset_index()
            vader_counts.columns = ['Sentiment', 'Count']
            fig = px.pie(vader_counts, names='Sentiment', values='Count', title='Sentiment Distribution (VADER)')
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Bar Chart: Sentiment by Company
            st.markdown('<div class="section-container">', unsafe_allow_html=True)
            st.subheader(" Sentiment by Company (VADER)")
            company_sentiment = filtered_df.groupby(['Company', 'Sentiment_VADER']).size().reset_index(name='Count')
            fig = px.bar(company_sentiment, x='Company', y='Count', color='Sentiment_VADER', barmode='group')
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Monthly breakdown
            st.markdown('<div class="section-container">', unsafe_allow_html=True)
            st.subheader(" Monthly Sentiment Breakdown")
            month_summary = filtered_df.groupby(['Month', 'Sentiment_VADER']).size().unstack().fillna(0)
            st.dataframe(month_summary, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Monthly Stacked Bar Chart
            st.markdown('<div class="section-container">', unsafe_allow_html=True)
            st.subheader(" Monthly Sentiment Bar Chart")
            fig = px.bar(month_summary, x=month_summary.index, y=month_summary.columns,
                        title="Monthly VADER Sentiment", labels={'value': 'Count', 'Month': 'Month'})
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Trend chart
            st.markdown('<div class="section-container">', unsafe_allow_html=True)
            st.subheader(" Sentiment Trend Over Time")
            trend_data = filtered_df.groupby([pd.Grouper(key='ReviewDate', freq='D'), 'Sentiment_VADER']).size().unstack().fillna(0)
            st.line_chart(trend_data, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Word cloud
            st.markdown('<div class="section-container">', unsafe_allow_html=True)
            st.subheader(" Word Cloud")
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
            st.markdown('</div>', unsafe_allow_html=True)

            # Interactive Explorer
            st.markdown('<div class="section-container">', unsafe_allow_html=True)
            st.subheader("🔍 Explore Reviews by Sentiment and Keyword")
            selected_sentiment = st.selectbox("Filter by VADER Sentiment", ["All", "Positive", "Negative", "Neutral"])
            keyword = st.text_input("Search keyword in reviews")
            explore_df = filtered_df
            if selected_sentiment != "All":
                explore_df = explore_df[explore_df['Sentiment_VADER'] == selected_sentiment]
            if keyword:
                explore_df = explore_df[explore_df[review_col].str.contains(keyword, case=False)]
            st.write(f"Found {len(explore_df)} matching reviews:")
            st.dataframe(explore_df[[review_col, 'Sentiment_VADER', 'ReviewDate', 'Company']].head(50), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error processing CSV: {str(e)}")