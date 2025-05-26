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
from collections import Counter
import zipfile

# Setup
st.set_page_config(page_title="Sentiment Insights", layout="wide")
analyzer = SentimentIntensityAnalyzer()
bert_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
stopwords = set(STOPWORDS)

# Theme management
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

# Initialize session state for tracking suggestions
if 'past_suggestions' not in st.session_state:
    st.session_state.past_suggestions = []
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

# Define CSS for light and dark modes
light_mode_css = """
<style>
    .main {
        background-color: #f9fafb;
        color: #1a1a1a;
        padding: 20px;
        min-height: 100vh;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    .stSidebar {
        background-color: #f7f7f7;
        padding: 20px;
        border-right: 1px solid #e0e0e0;
        box-shadow: 2px 0 8px rgba(0, 0, 0, 0.05);
    }
    .stSidebar h2 {
        color: #1e40af;
        font-size: 20px;
        font-weight: 600;
        margin-bottom: 15px;
    }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    .stSelectbox, .stFileUploader {
        background-color: #ffffff;
        color: #1a1a1a;
        border: 1px solid #d1d5db;
        border-radius: 8px;
        padding: 10px;
        transition: all 0.3s ease;
    }
    .stSelectbox:hover, .stFileUploader:hover {
        border-color: #2563eb;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
    }
    h1, h2, h3, h4, h5, h6 {
        color: #0f172a;
        font-weight: 600;
        margin-bottom: 20px;
    }
    .download-container {
        position: fixed;
        top: 20px;
        right: 100px;
        z-index: 1000;
        display: flex;
        justify-content: flex-end;
    }
    .download-container .stButton>button {
        background-color: #10b981;
        color: white;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 500;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    .download-container .stButton>button:hover {
        background-color: #059669;
        transform: translateY(-2px);
    }
    .individual-download-container {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 20px;
    }
    .individual-download-container .stButton>button {
        background-color: #10b981;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    .individual-download-container .stButton>button:hover {
        background-color: #059669;
        transform: translateY(-2px);
    }
    .theme-toggle-container {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 20px;
    }
    .theme-toggle-container .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 500;
        border: none;
        transition: all 0.3s ease;
    }
    .theme-toggle-container .stButton>button:hover {
        background-color: #1d4ed8;
        transform: translateY(-2px);
    }
    .section-container {
        margin-bottom: 25px;
        padding: 20px;
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease;
    }
    .section-container:hover {
        transform: translateY(-3px);
    }
    .section-container h3 {
        font-size: 18px;
        color: #1e40af;
        margin-bottom: 15px;
    }
    .how-it-works {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 20px;
        margin-top: 20px;
    }
    .how-it-works .stColumn {
        padding: 15px;
        background-color: #f9fafb;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        transition: transform 0.2s ease;
    }
    .how-it-works .stColumn:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    .stFileUploader > div > div {
        border: 2px dashed #d1d5db;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        transition: border-color 0.3s ease;
    }
    .stFileUploader > div > div:hover {
        border-color: #2563eb;
    }
    .stFileUploader > div > div p {
        color: #0f172a !important;
    }
    .footer {
        background-color: #2563eb;
        color: white;
        padding: 10px;
        text-align: center;
        border-radius: 8px;
        margin-top: 30px;
        font-size: 14px;
    }
    .loading-spinner {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100px;
    }
    .loading-spinner::before {
        content: '';
        width: 40px;
        height: 40px;
        border: 4px solid #2563eb;
        border-top: 4px solid transparent;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    @media (max-width: 768px) {
        .download-container {
            position: static;
            margin: 20px 0;
            justify-content: center;
        }
        .individual-download-container {
            justify-content: center;
        }
        .theme-toggle-container {
            justify-content: center;
        }
        .how-it-works {
            grid-template-columns: 1fr;
        }
        .main {
            padding: 15px;
        }
    }
</style>
"""

dark_mode_css = """
<style>
    .main {
        background-color: #2d3748;
        color: #e5e7eb;
        padding: 20px;
        min-height: 100vh;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    .stSidebar {
        background-color: #111827;
        padding: 20px;
        border-right: 1px solid #374151;
        box-shadow: 2px 0 8px rgba(0, 0, 0, 0.3);
    }
    .stSidebar h2 {
        color: #60a5fa;
        font-size: 20px;
        font-weight: 600;
        margin-bottom: 15px;
    }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    .stSelectbox, .stFileUploader {
        background-color: #374151;
        color: #e5e7eb;
        border: 1px solid #4b5563;
        border-radius: 8px;
        padding: 10px;
        transition: all 0.3s ease;
    }
    .stSelectbox:hover, .stFileUploader:hover {
        border-color: #60a5fa;
        box-shadow: 0 0 0 3px rgba(96, 165, 250, 0.1);
    }
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff;
        font-weight: 600;
        margin-bottom: 20px;
    }
    .download-container {
        position: fixed;
        top: 20px;
        right: 100px;
        z-index: 1000;
        display: flex;
        justify-content: flex-end;
    }
    .download-container .stButton>button {
        background-color: #10b981;
        color: white;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 500;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    .download-container .stButton>button:hover {
        background-color: #059669;
        transform: translateY(-2px);
    }
    .individual-download-container {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 20px;
    }
    .individual-download-container .stButton>button {
        background-color: #10b981;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    .individual-download-container .stButton>button:hover {
        background-color: #059669;
        transform: translateY(-2px);
    }
    .theme-toggle-container {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 20px;
    }
    .theme-toggle-container .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 500;
        border: none;
        transition: all 0.3s ease;
    }
    .theme-toggle-container .stButton>button:hover {
        background-color: #1d4ed8;
        transform: translateY(-2px);
    }
    .section-container {
        margin-bottom: 25px;
        padding: 20px;
        background-color: #374151;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease;
    }
    .section-container:hover {
        transform: translateY(-3px);
    }
    .section-container h3 {
        font-size: 18px;
        color: #60a5fa;
        margin-bottom: 15px;
    }
    .how-it-works {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 20px;
        margin-top: 20px;
    }
    .how-it-works .stColumn {
        padding: 15px;
        background-color: #4b5563;
        border-radius: 10px;
        border: 1px solid #6b7280;
        transition: transform 0.2s ease;
    }
    .how-it-works .stColumn:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    .stFileUploader > div > div {
        border: 2px dashed #6b7280;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        transition: border-color 0.3s ease;
    }
    .stFileUploader > div > div:hover {
        border-color: #60a5fa;
    }
    .footer {
        background-color: #1e40af;
        color: white;
        padding: 10px;
        text-align: center;
        border-radius: 8px;
        margin-top: 30px;
        font-size: 14px;
    }
    .loading-spinner {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100px;
    }
    .loading-spinner::before {
        content: '';
        width: 40px;
        height: 40px;
        border: 4px solid #60a5fa;
        border-top: 4px solid transparent;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    @media (max-width: 768px) {
        .download-container {
            position: static;
            margin: 20px 0;
            justify-content: center;
        }
        .individual-download-container {
            justify-content: center;
        }
        .theme-toggle-container {
            justify-content: center;
        }
        .how-it-works {
            grid-template-columns: 1fr;
        }
        .main {
            padding: 15px;
        }
    }
</style>
"""

# Apply theme
def apply_theme():
    if st.session_state.theme == 'dark':
        st.markdown(dark_mode_css, unsafe_allow_html=True)
    else:
        st.markdown(light_mode_css, unsafe_allow_html=True)

apply_theme()

# Theme toggle
col1, col2 = st.columns([10, 1])
with col2:
    st.markdown('<div class="theme-toggle-container">', unsafe_allow_html=True)
    theme_button = st.button("🌙" if st.session_state.theme == 'light' else "☀")
    if theme_button:
        st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# Plan selection
st.sidebar.header(" Settings")
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

page = st.sidebar.selectbox(" Navigate", list(PAGES.keys()), key="page_select", on_change=lambda: setattr(st.session_state, 'page', st.session_state.page_select))

# Sentiment analysis function
@st.cache_data
def analyze_sentiments(reviews):
    def analyze_sentiment(text):
        if not isinstance(text, str) or not text.strip():
            return {
                'VADER': 'Neutral', 'TextBlob': 'Neutral', 'BERT': 'Neutral',
                'VADER_Score': 0.0, 'TextBlob_Score': 0.0, 'BERT_Score': 0.5
            }
        vader_score = analyzer.polarity_scores(text)
        vader_sentiment = 'Positive' if vader_score['compound'] >= 0.05 else 'Negative' if vader_score['compound'] <= -0.05 else 'Neutral'
        try:
            blob = TextBlob(text)
            blob_sentiment = 'Positive' if blob.sentiment.polarity > 0 else 'Negative' if blob.sentiment.polarity < 0 else 'Neutral'
            blob_score = blob.sentiment.polarity
        except:
            blob_sentiment, blob_score = 'Neutral', 0.0
        bert_result = bert_analyzer(text[:512])[0]
        bert_sentiment = 'Positive' if bert_result['label'] == 'POSITIVE' else 'Negative'
        return {
            'VADER': vader_sentiment, 'TextBlob': blob_sentiment, 'BERT': bert_sentiment,
            'VADER_Score': vader_score['compound'], 'TextBlob_Score': blob_score, 'BERT_Score': bert_result['score']
        }
    return [analyze_sentiment(text) for text in reviews]

# Feedback improvement suggestions function
def get_top_suggestion(reviews, sentiment, sentiment_counts, sentiment_scores, review_dates, industry="generic"):
    suggestion_map = {
        "e-commerce": {
            "shipping": "Optimize shipping processes by partnering with reliable carriers or offering faster delivery options to improve customer satisfaction.",
            "product": "Enhance product quality control by implementing stricter checks or updating product descriptions to set clearer expectations.",
            "return": "Streamline the return process by automating approvals or extending the return window to build customer trust.",
            "customer service": "Improve customer service by offering 24/7 support or training staff to handle complaints more empathetically.",
            "price": "Reassess pricing strategy by introducing competitive discounts or loyalty programs to enhance perceived value.",
            "website": "Enhance website usability by improving navigation or reducing checkout steps to boost conversion rates.",
            "delivery": "Reduce delivery delays by optimizing logistics or providing real-time tracking updates to customers."
        },
        "restaurants": {
            "service": "Enhance service quality by training staff on customer interaction or increasing staff during peak hours to reduce wait times.",
            "food": "Improve food quality by sourcing fresher ingredients or ensuring consistency in preparation to meet customer expectations.",
            "ambiance": "Upgrade restaurant ambiance by investing in better lighting, seating, or music to create a more welcoming environment.",
            "price": "Adjust pricing strategy by offering value-for-money deals or combo meals to improve customer satisfaction.",
            "cleanliness": "Implement stricter hygiene protocols, such as regular deep cleaning, to address concerns and improve customer trust.",
            "menu": "Diversify the menu by adding trending dishes or catering to dietary preferences like vegan or gluten-free options.",
            "wait": "Reduce wait times by optimizing table turnover or implementing a reservation system to manage peak hours."
        },
        "service providers": {
            "response": "Speed up response times by implementing automated ticketing systems or increasing support staff availability.",
            "support": "Enhance support quality by providing specialized training or offering multi-channel support (e.g., chat, email, phone).",
            "booking": "Simplify booking by introducing an intuitive online portal or mobile app for seamless scheduling.",
            "quality": "Ensure consistent service quality by conducting regular audits or gathering post-service feedback to identify gaps.",
            "communication": "Improve communication by sending proactive updates or ensuring clarity in service agreements.",
            "availability": "Increase service availability by extending operating hours or offering emergency on-call support.",
            "professionalism": "Boost professionalism by standardizing staff uniforms or improving interpersonal skills through training."
        },
        "generic": {
            "service": "Enhance overall service quality by conducting regular training or implementing a feedback loop to address issues promptly.",
            "product": "Maintain high product standards by sourcing better materials or conducting regular quality checks.",
            "support": "Strengthen customer support by reducing response times or offering personalized assistance.",
            "experience": "Improve customer experience by personalizing interactions or simplifying processes.",
            "feedback": "Actively collect feedback through surveys and act on it to show customers their opinions matter.",
            "reliability": "Ensure reliability by meeting deadlines consistently or improving product durability.",
            "value": "Increase perceived value by offering bundled services or exclusive perks for loyal customers."
        }
    }

    if not reviews:
        return f"No {sentiment.lower()} feedback available for analysis."

    # Calculate sentiment distribution and average sentiment score
    total_reviews = sum(sentiment_counts.values())
    sentiment_dist = {s: count / total_reviews * 100 for s, count in sentiment_counts.items()}
    avg_sentiment_score = sum(sentiment_scores[sentiment]) / len(sentiment_scores[sentiment]) if sentiment_scores[sentiment] else 0

    # Aggregate keywords and calculate their prevalence
    all_keywords = []
    keyword_reviews = []
    for review in reviews:
        try:
            words = TextBlob(review.lower()).words
            keywords = [word for word in words if word not in stopwords and len(word) > 3]
            all_keywords.extend(keywords)
            keyword_reviews.append((keywords, review))
        except:
            continue

    if not all_keywords:
        return "Insufficient meaningful keywords found. Review feedback details and consider general improvements."

    keyword_counts = Counter(all_keywords)
    total_keywords = sum(keyword_counts.values())
    top_keywords = [(keyword, count / total_keywords * 100) for keyword, count in keyword_counts.most_common(3)]

    # Check for recent trends (last 30 days) if date column is available
    recent_trend = ""
    if review_dates:
        recent_date = pd.Timestamp('2025-05-23')  # Current date
        thirty_days_ago = recent_date - pd.Timedelta(days=30)
        recent_reviews = [review for review, date in zip(reviews, review_dates) if pd.notna(date) and thirty_days_ago <= date <= recent_date]
        if recent_reviews:
            recent_keywords = []
            for review in recent_reviews:
                try:
                    words = TextBlob(review.lower()).words
                    keywords = [word for word in words if word not in stopwords and len(word) > 3]
                    recent_keywords.extend(keywords)
                except:
                    continue
            if recent_keywords:
                recent_keyword_counts = Counter(recent_keywords)
                top_recent_keyword = recent_keyword_counts.most_common(1)[0][0] if recent_keyword_counts else None
                if top_recent_keyword:
                    recent_trend = f" Notably, in the last 30 days, '{top_recent_keyword}' has emerged as a frequent concern."

    # Get industry-specific suggestions
    industry_suggestions = suggestion_map.get(industry, suggestion_map["generic"])

    # Map top keywords to suggestion
    base_suggestion = None
    matched_keywords = []
    for keyword, _ in top_keywords:
        for industry_key, suggestion in industry_suggestions.items():
            if industry_key in keyword:
                base_suggestion = suggestion
                matched_keywords.append(keyword)
                break
        if len(matched_keywords) >= 2:  # Use up to 2 keywords for a more comprehensive suggestion
            break

    # Fallback if no keyword matches
    if not base_suggestion:
        if sentiment == "Positive":
            base_suggestion = "Continue focusing on overall customer satisfaction to maintain positive feedback."
        elif sentiment == "Neutral":
            base_suggestion = "Explore opportunities to enhance customer experience and convert neutral feedback into positive."
        else:  # Negative
            base_suggestion = "Address common feedback themes to improve customer satisfaction and reduce negative sentiment."
        matched_keywords = [top_keywords[0][0]] if top_keywords else ["general feedback"]

    # Calculate keyword prevalence
    keyword_prevalence = ", ".join([f"'{kw}' ({pct:.1f}% of keywords)" for kw, pct in top_keywords if kw in matched_keywords])

    # Assess sentiment intensity
    sentiment_intensity = "strong" if abs(avg_sentiment_score) > 0.5 else "moderate" if abs(avg_sentiment_score) > 0.2 else "mild"

    # Construct detailed suggestion
    sentiment_percentage = sentiment_dist.get(sentiment, 0)
    detailed_suggestion = (
        f"Based on {sentiment_percentage:.1f}% of reviews being {sentiment.lower()} with a {sentiment_intensity} sentiment score (avg: {avg_sentiment_score:.2f}), "
        f"key issues include {keyword_prevalence}.{recent_trend} {base_suggestion}"
    )

    # Ensure uniqueness by checking against past suggestions
    counter = 1
    original_suggestion = detailed_suggestion
    while detailed_suggestion in st.session_state.past_suggestions:
        next_keyword = top_keywords[min(counter, len(top_keywords)-1)][0]
        detailed_suggestion = (
            f"{original_suggestion} Additionally, consider addressing '{next_keyword}' to further improve {sentiment.lower()} feedback trends."
        )
        counter += 1

    # Add to past suggestions
    st.session_state.past_suggestions.append(detailed_suggestion)

    return detailed_suggestion

# Process single file function
def process_file(uploaded_file, file_name):
    try:
        with st.spinner(f"Processing {file_name}..."):
            # Load and process the CSV
            df = pd.read_csv(uploaded_file)
            columns = df.columns.tolist()
            st.subheader(f"Column Selection for {file_name}")
            review_col = st.selectbox(f"Select Review Text Column ({file_name})", options=columns, key=f"review_col_{file_name}")
            date_col = st.selectbox(f"Select Date Column (optional) ({file_name})", options=["None"] + columns, key=f"date_col_{file_name}")
            company_col = st.selectbox(f"Select Company Column (optional) ({file_name})", options=["None"] + columns, key=f"company_col_{file_name}")

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
                st.warning(f"Your {plan} allows only {MAX_REVIEWS} reviews for {file_name}. Processing the first {MAX_REVIEWS} reviews.")
                df = df.head(MAX_REVIEWS)
                time.sleep(2)

            # Sentiment analysis
            df['Sentiment'] = analyze_sentiments(df[review_col].tolist())
            df['Sentiment_VADER'] = df['Sentiment'].apply(lambda x: x['VADER'])
            df['Sentiment_TextBlob'] = df['Sentiment'].apply(lambda x: x['TextBlob'])
            df['Sentiment_BERT'] = df['Sentiment'].apply(lambda x: x['BERT'])
            df['Month'] = df['ReviewDate'].dt.to_period('M').astype(str)

            # Collect sentiment scores for each category
            sentiment_scores = {'Positive': [], 'Negative': [], 'Neutral': []}
            for sentiment_dict in df['Sentiment']:
                vader_sentiment = sentiment_dict['VADER']
                vader_score = sentiment_dict['VADER_Score']
                sentiment_scores[vader_sentiment].append(vader_score)

            # Company filter
            company_filter = st.selectbox(f"Select Company ({file_name})", ["All"] + df['Company'].unique().tolist(), key=f"company_filter_{file_name}")
            filtered_df = df if company_filter == "All" else df[df['Company'] == company_filter]

            # Download report
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                filtered_df.to_excel(writer, index=False, sheet_name='ReviewData')
                filtered_df['Sentiment_VADER'].value_counts().to_frame(name='Count').to_excel(writer, sheet_name='SentimentCounts')
                month_summary = filtered_df.groupby(['Month', 'Sentiment_VADER']).size().unstack().fillna(0)
                month_summary.to_excel(writer, sheet_name='MonthlyBreakdown')
            report_data = output.getvalue()

            # Add download button for this specific file's report before the Review List section
            st.markdown('<div class="individual-download-container">', unsafe_allow_html=True)
            st.download_button(
                label=f"Download Report for {file_name}",
                data=report_data,
                file_name=f"sentiment_report_{file_name}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"download_{file_name}"
            )
            st.markdown('</div>', unsafe_allow_html=True)

            # Review preview
            st.markdown('<div class="section-container">', unsafe_allow_html=True)
            st.subheader(f"Review List with Sentiment ({file_name})")
            st.dataframe(filtered_df[[review_col, 'Sentiment_VADER', 'Sentiment_TextBlob', 'Sentiment_BERT', 'ReviewDate', 'Company']].head(20))
            st.markdown('</div>', unsafe_allow_html=True)

            # Sentiment counts
            st.markdown('<div class="section-container">', unsafe_allow_html=True)
            st.subheader(f"Sentiment Count Summary ({file_name})")
            st.write("*VADER Sentiment Counts:*")
            st.write(filtered_df['Sentiment_VADER'].value_counts())
            st.write("*TextBlob Sentiment Counts:*")
            st.write(filtered_df['Sentiment_TextBlob'].value_counts())
            st.write("*BERT Sentiment Counts:*")
            st.write(filtered_df['Sentiment_BERT'].value_counts())
            st.markdown('</div>', unsafe_allow_html=True)

            # Pie Chart
            st.markdown('<div class="section-container">', unsafe_allow_html=True)
            st.subheader(f"VADER Sentiment Pie Chart ({file_name})")
            vader_counts = filtered_df['Sentiment_VADER'].value_counts().reset_index()
            vader_counts.columns = ['Sentiment', 'Count']
            fig = px.pie(vader_counts, names='Sentiment', values='Count', title=f'Sentiment Distribution (VADER) - {file_name}')
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Bar Chart: Sentiment by Company
            st.markdown('<div class="section-container">', unsafe_allow_html=True)
            st.subheader(f"Sentiment by Company (VADER) ({file_name})")
            company_sentiment = filtered_df.groupby(['Company', 'Sentiment_VADER']).size().reset_index(name='Count')
            fig = px.bar(company_sentiment, x='Company', y='Count', color='Sentiment_VADER', barmode='group')
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Monthly breakdown
            st.markdown('<div class="section-container">', unsafe_allow_html=True)
            st.subheader(f"Monthly Sentiment Breakdown ({file_name})")
            month_summary = filtered_df.groupby(['Month', 'Sentiment_VADER']).size().unstack().fillna(0)
            st.dataframe(month_summary, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Monthly Stacked Bar Chart
            st.markdown('<div class="section-container">', unsafe_allow_html=True)
            st.subheader(f"Monthly Sentiment Bar Chart ({file_name})")
            fig = px.bar(month_summary, x=month_summary.index, y=month_summary.columns,
                        title=f"Monthly VADER Sentiment ({file_name})", labels={'value': 'Count', 'Month': 'Month'})
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Trend chart
            st.markdown('<div class="section-container">', unsafe_allow_html=True)
            st.subheader(f"Sentiment Trend Over Time ({file_name})")
            trend_data = filtered_df.groupby([pd.Grouper(key='ReviewDate', freq='D'), 'Sentiment_VADER']).size().unstack().fillna(0)
            st.line_chart(trend_data, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Word cloud
            st.markdown('<div class="section-container">', unsafe_allow_html=True)
            st.subheader(f"Word Cloud ({file_name})")
            sentiment_filter = st.selectbox(f"Filter Word Cloud by Sentiment ({file_name})", ["All", "Positive", "Negative", "Neutral"], key=f"sentiment_filter_{file_name}")
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
                st.warning(f"No text available for the selected filters in {file_name}.")
            st.markdown('</div>', unsafe_allow_html=True)

            # Interactive Explorer
            st.markdown('<div class="section-container">', unsafe_allow_html=True)
            st.subheader(f"Explore Reviews by Sentiment and Keyword ({file_name})")
            selected_sentiment = st.selectbox(f"Filter by VADER Sentiment ({file_name})", ["All", "Positive", "Negative", "Neutral"], key=f"selected_sentiment_{file_name}")
            keyword = st.text_input(f"Search keyword in reviews ({file_name})", key=f"keyword_{file_name}")
            explore_df = filtered_df
            if selected_sentiment != "All":
                explore_df = explore_df[explore_df['Sentiment_VADER'] == selected_sentiment]
            if keyword:
                explore_df = explore_df[explore_df[review_col].str.contains(keyword, case=False, na=False)]
            st.write(f"Found {len(explore_df)} matching reviews in {file_name}:")
            st.dataframe(explore_df[[review_col, 'Sentiment_VADER', 'ReviewDate', 'Company']].head(50), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Feedback Improvement Suggestions
            st.markdown('<div class="section-container">', unsafe_allow_html=True)
            st.subheader(f"Feedback Improvement Suggestions ({file_name})")
            industry = st.selectbox(f"Select Industry for Suggestions ({file_name})", ["Generic", "E-commerce", "Restaurants", "Service Providers"], key=f"industry_{file_name}")
            sentiment_counts = filtered_df['Sentiment_VADER'].value_counts().to_dict()
            suggestions = []
            review_dates = filtered_df['ReviewDate'].tolist() if 'ReviewDate' in filtered_df else None
            for sentiment in ["Positive", "Negative", "Neutral"]:
                sentiment_reviews = filtered_df[filtered_df['Sentiment_VADER'] == sentiment][review_col].tolist()
                suggestion = get_top_suggestion(sentiment_reviews, sentiment, sentiment_counts, sentiment_scores, review_dates, industry.lower())
                suggestions.append({
                    "Sentiment": sentiment,
                    "Top Suggestion": suggestion,
                    "Review Count": len(sentiment_reviews)
                })
            suggestion_df = pd.DataFrame(suggestions)
            st.dataframe(suggestion_df, use_container_width=True)
            suggestion_output = io.BytesIO()
            with pd.ExcelWriter(suggestion_output, engine='openpyxl') as writer:
                suggestion_df.to_excel(writer, index=False, sheet_name='Top_Suggestions')

            # First 10 Rows Individual Suggestions
            st.markdown('<div class="section-container">', unsafe_allow_html=True)
            st.subheader(f"First 10 Rows Individual Feedback Improvement Suggestions ({file_name})")
            first_ten_df = filtered_df.head(10)
            individual_suggestions = []
            for index, row in first_ten_df.iterrows():
                review = row[review_col]
                vader_sentiment = row['Sentiment_VADER']
                date = row['ReviewDate'] if pd.notna(row['ReviewDate']) else None
                single_review_scores = {vader_sentiment: [row['Sentiment']['VADER_Score']]}
                single_review_counts = {vader_sentiment: 1}
                single_review_dates = [date] if date else None
                suggestion = get_top_suggestion([review], vader_sentiment, single_review_counts, single_review_scores, single_review_dates, industry.lower())
                individual_suggestions.append({
                    "Review": review,
                    "Sentiment": vader_sentiment,
                    "Suggestion": suggestion
                })
            individual_suggestion_df = pd.DataFrame(individual_suggestions)
            st.dataframe(individual_suggestion_df, use_container_width=True)
            individual_suggestion_output = io.BytesIO()
            with pd.ExcelWriter(individual_suggestion_output, engine='openpyxl') as writer:
                individual_suggestion_df.to_excel(writer, index=False, sheet_name='First_Ten_Individual_Suggestions')

            return filtered_df, suggestion_df, report_data, suggestion_output.getvalue(), individual_suggestion_output.getvalue()

    except Exception as e:
        st.error(f"Error processing {file_name}: {str(e)}")
        return None, None, None, None, None

# Page 1: Home
if page == "Home":
    st.title("Analyze Customer Sentiment Instantly")
    st.write("Upload multiple customer review CSVs to unlock AI-powered sentiment analysis and actionable insights for each file.")
    
    # File upload section
    st.markdown("""
    <div style='border: 2px dashed #d1d5db; padding: 20px; text-align: center; border-radius: 10px;'>
        <p style='font-weight: 500; color: #0f172a;'>Drag & drop your review files here</p>
        <p style='color: #4b5563;'>Multiple CSVs supported, up to 10MB each.</p>
    </div>
    """, unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Browse Files", type=["csv"], accept_multiple_files=True)
    
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        st.session_state.page = "Dashboard"
        st.rerun()
    
    # How It Works section
    st.subheader("How It Works")
    st.markdown('<div class="how-it-works">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(" Upload Your Data")
        st.write("Easily drag and drop or select multiple CSV files.")
    with col2:
        st.markdown(" Fast Analysis")
        st.write("Our AI engine processes each file to identify sentiment.")
    with col3:
        st.markdown(" Visualize Insights")
        st.write("View charts and summaries for each file's sentiment trends.")
    with col4:
        st.markdown(" Detailed Review Table")
        st.write("Explore individual reviews with sentiment labels per file.")
    st.markdown('</div>', unsafe_allow_html=True)

# Page 2: Dashboard
elif page == "Dashboard":
    st.title("AI Review Sentiment Dashboard")
    st.write("Upload multiple review CSVs to analyze sentiments, view trends, and generate reports for each file.")

    # File upload
    uploaded_files = st.session_state.uploaded_files if st.session_state.uploaded_files else st.file_uploader("Upload your CSV files", type=["csv"], accept_multiple_files=True)
    
    if uploaded_files:
        # Create a ZIP file for downloading all reports
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Create tabs for each file
            tabs = st.tabs([file.name for file in uploaded_files])
            all_dfs = []
            all_suggestion_dfs = []
            
            for i, uploaded_file in enumerate(uploaded_files):
                with tabs[i]:
                    st.subheader(f"Analysis for {uploaded_file.name}")
                    filtered_df, suggestion_df, report_data, suggestion_data, individual_suggestion_data = process_file(uploaded_file, uploaded_file.name)
                    if filtered_df is not None:
                        all_dfs.append((uploaded_file.name, filtered_df))
                        all_suggestion_dfs.append((uploaded_file.name, suggestion_df))
                        # Add individual report to ZIP
                        zip_file.writestr(f"sentiment_report_{uploaded_file.name}.xlsx", report_data)
                        # Add suggestions report to ZIP
                        zip_file.writestr(f"top_suggestions_report_{uploaded_file.name}.xlsx", suggestion_data)
                        # Add individual suggestions report to ZIP
                        zip_file.writestr(f"first_ten_individual_suggestions_{uploaded_file.name}.xlsx", individual_suggestion_data)
        
        # Download all reports as ZIP
        st.markdown('<div class="download-container">', unsafe_allow_html=True)
        st.download_button(
            label="Download All Reports as ZIP",
            data=zip_buffer.getvalue(),
            file_name="all_sentiment_reports.zip",
            mime="application/zip",
            key="download_all"
        )
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    © 2025 Sentiment Insights | Copyright S Kajamalan @Detech
</div>
""", unsafe_allow_html=True)