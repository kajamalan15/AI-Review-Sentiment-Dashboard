import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import time
import io
from collections import Counter
import zipfile

# Error handling for transformers import
try:
    from transformers import pipeline
except ImportError as e:
    st.error(f"Failed to import transformers: {str(e)}. Please check your dependencies.")
    st.stop()

# Setup
st.set_page_config(page_title="Sentiment Insights", layout="wide")
analyzer = SentimentIntensityAnalyzer()
@st.cache_resource
def load_bert_analyzer():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
bert_analyzer = load_bert_analyzer()
stopwords = set(STOPWORDS)

# Theme management
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

# Initialize session state
if 'past_suggestions' not in st.session_state:
    st.session_state.past_suggestions = []
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'goal' not in st.session_state:
    st.session_state.goal = "None"
if 'edited_reviews' not in st.session_state:
    st.session_state.edited_reviews = {}

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
    .stSelectbox, .stFileUploader, .stTextInput {
        background-color: #ffffff;
        color: #1a1a1a;
        border: 1px solid #d1d5db;
        border-radius: 8px;
        padding: 10px;
        transition: all 0.3s ease;
    }
    .stSelectbox:hover, .stFileUploader:hover, .stTextInput:hover {
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
    .end-process-container {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 20px;
    }
    .end-process-container .stButton>button {
        background-color: #ef4444;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    .end-process-container .stButton>button:hover {
        background-color: #dc2626;
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
    .goal-container {
        margin-bottom: 25px;
        padding: 20px;
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease;
    }
    .goal-container:hover {
        transform: translateY(-3px);
    }
    .goal-container h3 {
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
        .end-process-container {
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
    .stSelectbox, .stFileUploader, .stTextInput {
        background-color: #374151;
        color: #e5e7eb;
        border: 1px solid #4b5563;
        border-radius: 8px;
        padding: 10px;
        transition: all 0.3s ease;
    }
    .stSelectbox:hover, .stFileUploader:hover, .stTextInput:hover {
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
    .end-process-container {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 20px;
    }
    .end-process-container .stButton>button {
        background-color: #ef4444;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    .end-process-container .stButton>button:hover {
        background-color: #dc2626;
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
    .goal-container {
        margin-bottom: 25px;
        padding: 20px;
        background-color: #374151;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease;
    }
    .goal-container:hover {
        transform: translateY(-3px);
    }
    .goal-container h3 {
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
        .end-process-container {
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

# Sidebar with Plan Selection and Navigation
st.sidebar.header("Settings")
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

page = st.sidebar.selectbox("Navigate", list(PAGES.keys()), key="page_select", on_change=lambda: setattr(st.session_state, 'page', st.session_state.page_select))

# Function to reset session state and restart
def reset_app():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.theme = 'light'
    st.session_state.past_suggestions = []
    st.session_state.uploaded_files = []
    st.session_state.goal = "None"
    st.session_state.edited_reviews = {}
    st.session_state.page = "Home"
    st.rerun()

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

    # Aggregate keywords
    all_keywords = []
    for review in reviews:
        try:
            words = TextBlob(review.lower()).words
            keywords = [word for word in words if word not in stopwords and len(word) > 3]
            all_keywords.extend(keywords)
        except:
            continue

    if not all_keywords:
        return "Insufficient meaningful keywords found. Review feedback details and consider general improvements."

    keyword_counts = Counter(all_keywords)
    total_keywords = sum(keyword_counts.values())
    top_keywords = [(keyword, count / total_keywords * 100) for keyword, count in keyword_counts.most_common(3)]

    # Check for recent trends (last 30 days)
    recent_trend = ""
    if review_dates:
        recent_date = pd.Timestamp('2025-05-29')
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
        if len(matched_keywords) >= 2:
            break

    # Incorporate user goal
    goal = st.session_state.goal.lower()
    goal_prefix = ""
    if goal != "none":
        if "reduce negative feedback" in goal and sentiment == "Negative":
            goal_prefix = f"To align with your goal to {goal}, prioritize: "
        elif "increase positive feedback" in goal and sentiment == "Positive":
            goal_prefix = f"To align with your goal to {goal}, continue to: "
        elif "improve neutral feedback" in goal and sentiment == "Neutral":
            goal_prefix = f"To align with your goal to {goal}, focus on: "
        elif "enhance customer satisfaction" in goal:
            goal_prefix = f"To align with your goal to {goal}, emphasize: "
        elif "custom" in goal:
            goal_prefix = f"To align with your custom goal '{goal}', consider: "

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
        f"{goal_prefix}Based on {sentiment_percentage:.1f}% of reviews being {sentiment.lower()} with a {sentiment_intensity} sentiment score (avg: {avg_sentiment_score:.2f}), "
        f"key issues include {keyword_prevalence}.{recent_trend} {base_suggestion}"
    )

    # Ensure uniqueness
    counter = 1
    original_suggestion = detailed_suggestion
    while detailed_suggestion in st.session_state.past_suggestions:
        next_keyword = top_keywords[min(counter, len(top_keywords)-1)][0]
        detailed_suggestion = (
            f"{original_suggestion} Additionally, consider addressing '{next_keyword}' to further improve {sentiment.lower()} feedback trends."
        )
        counter += 1

    st.session_state.past_suggestions.append(detailed_suggestion)
    return detailed_suggestion

# Function to generate a final suggestion based on the goal
def get_final_goal_suggestion(goal):
    goal = goal.lower()
    suggestion_map = {
        "reduce shipping complaints by 20%": "Partner with reliable carriers to ensure faster and more dependable deliveries.",
        "increase positive product reviews by 15%": "Enhance product quality and provide clear descriptions to boost customer satisfaction.",
        "convert neutral feedback to positive": "Engage neutral customers with follow-up offers to turn their feedback positive.",
        "improve customer service responsiveness": "Train staff for empathetic responses and offer 24/7 support channels.",
        "enhance checkout process satisfaction": "Simplify checkout with fewer steps and clearer payment options.",
        "custom": f"Focus on key areas of your custom goal '{goal}' to drive targeted improvements."
    }
    return suggestion_map.get(goal, f"Align actions with your goal '{goal}' to improve overall customer experience.")

# Process single file function
def process_file(uploaded_file, file_name):
    if uploaded_file.size > 10_000_000:  # 10MB limit
        st.error(f"{file_name} exceeds 10MB. Please upload a smaller file.")
        return None, None, None, None
    try:
        with st.spinner(f"Processing {file_name}..."):
            progress_bar = st.progress(0)
            # Load and process the CSV
            df = pd.read_csv(uploaded_file)
            progress_bar.progress(20)
            if df.empty:
                st.error(f"{file_name} is empty. Please upload a valid CSV file.")
                return None, None, None, None
            columns = df.columns.tolist()
            st.subheader(f"Column Selection for {file_name}")
            review_col = st.selectbox(f"Select Review Text Column ({file_name})", options=columns, key=f"review_col_{file_name}")
            date_col = st.selectbox(f"Select Date Column (optional) ({file_name})", options=["None"] + columns, key=f"date_col_{file_name}")
            company_col = st.selectbox(f"Select Company Column (optional) ({file_name})", options=["None"] + columns, key=f"company_col_{file_name}")

            # Clean data
            df = df.dropna(subset=[review_col])
            if df.empty:
                st.error(f"No valid reviews found in column '{review_col}' for {file_name} after cleaning.")
                return None, None, None, None
            df[review_col] = df[review_col].astype(str)
            if date_col != "None":
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                df['ReviewDate'] = df[date_col]
            else:
                df['ReviewDate'] = pd.date_range(start="2025-01-01", periods=len(df))
            df['Company'] = df[company_col] if company_col != "None" else "All"
            progress_bar.progress(40)

            # Apply review limit
            if len(df) > MAX_REVIEWS:
                st.warning(f"Your {plan} allows only {MAX_REVIEWS} reviews for {file_name}. Processing the first {MAX_REVIEWS} reviews.")
                df = df.head(MAX_REVIEWS)
                time.sleep(2)

            # Sentiment analysis
            try:
                df['Sentiment'] = analyze_sentiments(df[review_col].tolist())
                df['Sentiment_VADER'] = df['Sentiment'].apply(lambda x: x['VADER'])
                df['Sentiment_TextBlob'] = df['Sentiment'].apply(lambda x: x['TextBlob'])
                df['Sentiment_BERT'] = df['Sentiment'].apply(lambda x: x['BERT'])
            except Exception as e:
                st.error(f"Sentiment analysis failed for {file_name}: {str(e)}")
                return None, None, None, None
            progress_bar.progress(60)

            # Apply edited sentiments if available
            file_key = file_name
            if file_key in st.session_state.edited_reviews:
                for idx, sentiment in st.session_state.edited_reviews[file_key].items():
                    if idx in df.index:
                        df.at[idx, 'Sentiment_VADER'] = sentiment

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
            st.write(f"Debug: Filtered DataFrame size for {file_name} = {len(filtered_df)}")
            if filtered_df.empty:
                st.warning(f"No reviews match the selected company filter '{company_filter}' for {file_name}.")
                return None, None, None, None
            progress_bar.progress(80)

            # Download report
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                filtered_df.to_excel(writer, index=False, sheet_name='ReviewData')
                filtered_df['Sentiment_VADER'].value_counts().to_frame(name='Count').to_excel(writer, sheet_name='SentimentCounts')
                month_summary = filtered_df.groupby(['Month', 'Sentiment_VADER']).size().unstack().fillna(0)
                month_summary.to_excel(writer, sheet_name='MonthlyBreakdown')
            report_data = output.getvalue()

            # Add download button for this specific file's report
            st.markdown('<div class="individual-download-container">', unsafe_allow_html=True)
            st.download_button(
                label=f"Download Report for {file_name}",
                data=report_data,
                file_name=f"sentiment_report_{file_name}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"download_{file_name}"
            )
            st.markdown('</div>', unsafe_allow_html=True)

            # Editable Review Preview
            st.markdown('<div class="section-container">', unsafe_allow_html=True)
            st.subheader(f"Review List with Editable Sentiment ({file_name})")
            display_cols = [review_col, 'Sentiment_VADER', 'Sentiment_TextBlob', 'Sentiment_BERT', 'ReviewDate', 'Company']
            edit_df = filtered_df[display_cols].head(20).copy()
            edit_df['Sentiment_VADER'] = edit_df['Sentiment_VADER'].astype(str)
            edited_data = st.data_editor(
                edit_df,
                column_config={
                    'Sentiment_VADER': st.column_config.SelectboxColumn(
                        options=['Positive', 'Negative', 'Neutral'],
                        required=True
                    )
                },
                key=f"editor_{file_name}"
            )
            # Save edited sentiments
            if st.session_state[f"editor_{file_name}"]["edited_rows"]:
                if file_key not in st.session_state.edited_reviews:
                    st.session_state.edited_reviews[file_key] = {}
                for row_idx, changes in st.session_state[f"editor_{file_name}"]["edited_rows"].items():
                    if 'Sentiment_VADER' in changes:
                        global_idx = edit_df.index[row_idx]
                        st.session_state.edited_reviews[file_key][global_idx] = changes['Sentiment_VADER']
                        filtered_df.at[global_idx, 'Sentiment_VADER'] = changes['Sentiment_VADER']
            st.markdown('</div>', unsafe_allow_html=True)

            # Sentiment counts
            st.markdown('<div class="section-container">', unsafe_allow_html=True)
            st.subheader(f"Sentiment Count Summary ({file_name})")
            st.write("VADER Sentiment Counts:")
            st.write(filtered_df['Sentiment_VADER'].value_counts())
            st.write("TextBlob Sentiment Counts:")
            st.write(filtered_df['Sentiment_TextBlob'].value_counts())
            st.write("BERT Sentiment Counts:")
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
                        title=f"Monthly VADER Sentiment ({file_name})", labels={'value': 'Count', 'index': 'Month'})
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
            all_text = ' '.join(wc_df[review_col].tolist())
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
            selected_sentiment = st.selectbox(f"Filter by Sentiment ({file_name})", ["All", "Positive", "Negative", "Neutral"], key=f"selected_sentiment_{file_name}")
            keyword = st.text_input(f"Search keyword in reviews ({file_name})", key=f"keyword_{file_name}")
            explore_df = filtered_df
            if selected_sentiment != "All":
                explore_df = explore_df[explore_df['Sentiment_VADER'] == selected_sentiment]
            if keyword:
                explore_df = explore_df[explore_df[review_col].str.contains(keyword, case=False, na=False)]
            st.write(f"Found {len(explore_df)} matching reviews in {file_name}.")
            st.dataframe(explore_df[[review_col, 'Sentiment_VADER', 'ReviewDate', 'Company']].head(50), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Feedback Improvement Suggestions
            st.markdown('<div class="section-container">', unsafe_allow_html=True)
            st.subheader(f"Feedback Improvement ({file_name})")
            industry = st.selectbox(f"Select Industry for Suggestions ({file_name})", ["Generic", "E-commerce", "Restaurants", "Service Providers"], key=f"industry_{file_name}")
            sentiment_counts = filtered_df['Sentiment_VADER'].value_counts().to_dict()
            suggestions = []
            review_dates = filtered_df['ReviewDate'].tolist() if 'ReviewDate' in filtered_df else []
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
            progress_bar.progress(100)
            return filtered_df, suggestion_df, report_data, suggestion_output.getvalue()

    except Exception as e:
        st.error(f"Error processing {file_name}: {str(e)}")
        return None, None, None, None

# Function to generate an overall suggestion paragraph
def get_overall_suggestion(all_dfs, industry="generic"):
    if not all_dfs:
        return "No data available to generate an overall suggestion."

    # Combine all reviews and sentiments
    combined_reviews = []
    combined_sentiments = []
    combined_dates = []
    for _, df in all_dfs:
        if 'Review' in df.columns and 'Sentiment_VADER' in df.columns and 'ReviewDate' in df.columns:
            combined_reviews.extend(df['Review'].tolist())
            combined_sentiments.extend(df['Sentiment_VADER'].tolist())
            combined_dates.extend(df['ReviewDate'].tolist())

    if not combined_reviews:
        return "No valid reviews found across all files to generate an overall suggestion."

    # Calculate overall sentiment distribution
    sentiment_counts = Counter(combined_sentiments)
    total_reviews = len(combined_reviews)
    sentiment_dist = {s: count / total_reviews * 100 for s, count in sentiment_counts.items()}

    # Aggregate keywords across all reviews
    all_keywords = []
    for review in combined_reviews:
        try:
            words = TextBlob(review.lower()).words
            keywords = [word for word in words if word not in stopwords and len(word) > 3]
            all_keywords.extend(keywords)
        except:
            continue

    if not all_keywords:
        return "Insufficient meaningful keywords found across all files. Consider reviewing feedback details."

    keyword_counts = Counter(all_keywords)
    total_keywords = sum(keyword_counts.values())
    top_keywords = [(keyword, count / total_keywords * 100) for keyword, count in keyword_counts.most_common(3)]

    # Check for recent trends (last 30 days)
    recent_trend = ""
    if combined_dates:
        recent_date = pd.Timestamp('2025-05-29')
        thirty_days_ago = recent_date - pd.Timedelta(days=30)
        recent_reviews = [
            review for review, date in zip(combined_reviews, combined_dates)
            if pd.notna(date) and thirty_days_ago <= date <= recent_date
        ]
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

    # Determine dominant sentiment
    dominant_sentiment = max(sentiment_counts, key=sentiment_counts.get)
    sentiment_percentage = sentiment_dist.get(dominant_sentiment, 0)

    # Map top keywords to suggestion
    suggestion_map = {
        "e-commerce": {
            "shipping": "Optimize shipping processes across all operations to improve customer satisfaction.",
            "product": "Enhance product quality control to ensure consistency across all product lines.",
            "customer service": "Improve customer service by offering consistent, high-quality support across all channels."
        },
        "restaurants": {
            "service": "Enhance service quality across all locations by training staff to reduce wait times.",
            "food": "Improve food quality by ensuring consistency in preparation across all branches.",
            "cleanliness": "Implement stricter hygiene protocols across all locations to improve customer trust."
        },
        "service providers": {
            "response": "Speed up response times across all services by implementing automated systems.",
            "support": "Enhance support quality by providing specialized training across all teams.",
            "communication": "Improve communication by sending proactive updates across all customer interactions."
        },
        "generic": {
            "service": "Enhance overall service quality by implementing a consistent feedback loop.",
            "product": "Maintain high product standards by conducting regular quality checks.",
            "support": "Strengthen customer support by reducing response times across all interactions."
        }
    }
    industry_suggestions = suggestion_map.get(industry.lower(), suggestion_map["generic"])
    base_suggestion = None
    matched_keywords = []
    for keyword, _ in top_keywords:
        for industry_key, suggestion in industry_suggestions.items():
            if industry_key in keyword.lower():
                base_suggestion = suggestion
                matched_keywords.append(keyword)
                break
        if len(matched_keywords) >= 1:
            break

    if not base_suggestion:
        base_suggestion = "Focus on improving overall customer satisfaction by addressing common feedback themes."
        matched_keywords = [top_keywords[0][0]] if top_keywords else ["general feedback"]

    keyword_prevalence = ", ".join([f"'{kw}' ({pct:.1f}% of keywords)" for kw, pct in top_keywords if kw in matched_keywords])
    overall_suggestion = (
        f"Across all files, {sentiment_percentage:.1f}% of reviews are {dominant_sentiment.lower()}, with key themes including {keyword_prevalence}. "
        f"{recent_trend} {base_suggestion} This holistic approach can help align with your overall business goals."
    )
    return overall_suggestion

# Page 1: Home
if page == "Home":
    st.title("Analyze Customer Sentiment Instantly")
    st.write("Upload multiple e-commerce customer review CSVs to unlock AI-powered sentiment analysis and actionable insights for your online store.")

    # Goal Selection Section
    st.markdown('<div class="goal-container">', unsafe_allow_html=True)
    st.subheader("Define Your E-commerce Goal")
    st.write("Choose an e-commerce goal to optimize your online store and receive tailored improvement suggestions.")
    goal_options = [
        "None",
        "Reduce shipping complaints by 20%",
        "Increase positive product reviews by 15%",
        "Convert neutral feedback to positive",
        "Improve customer service responsiveness",
        "Enhance checkout process satisfaction",
        "Custom"
    ]
    selected_goal = st.selectbox("Select E-commerce Goal", options=goal_options, key="goal_select")
    if selected_goal == "Custom":
        custom_goal = st.text_input("Enter Custom E-commerce Goal", key="custom_goal")
        st.session_state.goal = custom_goal if custom_goal else "None"
    else:
        st.session_state.goal = selected_goal
    st.markdown('</div>', unsafe_allow_html=True)

    # File upload section
    st.markdown("""
    <div class="section-container">
        <div style='border: 2px dashed #d1d5db; padding: 20px; text-align: center; border-radius: 10px;'>
            <p style='font-weight: 500; color: #0f172a'>Drag & drop your e-commerce review files here</p>
            <p style='color: #4b5563'>Upload customer reviews from your online store (CSV, up to 10MB each).</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.warning("Upload CSV files under 10MB each for best performance.")
    uploaded_files = st.file_uploader("Upload E-commerce CSV files", type=["csv"], accept_multiple_files=True)

    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        st.session_state.page = "Dashboard"
        st.rerun()

    # How It Works
    st.subheader("How It Works")
    st.markdown('<div class="how-it-works">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("*Upload Data*")
        st.write('<p>Easily drag and drop or upload to select multiple CSV files.</p>', unsafe_allow_html=True)
    with col2:
        st.markdown("*Fast Analysis*")
        st.write('<p>Our AI engine processes each file to identify sentiment trends.</p>', unsafe_allow_html=True)
    with col3:
        st.markdown("*Visualize Insights*")
        st.write('<p>View charts and summaries for each file’s sentiment trends.</p>', unsafe_allow_html=True)
    with col4:
        st.markdown("*Detailed Review Table*")
        st.write('<p>Explore and edit individual reviews with sentiment labels per file.</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Page 2: Dashboard
elif page == "Dashboard":
    st.title("E-commerce AI Review Sentiment Dashboard")
    st.write("Upload multiple e-commerce review CSVs to analyze sentiments, view trends, and generate reports for each file.")

    # Display Goal Progress
    if st.session_state.goal != "None":
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.subheader("Goal Progress")
        st.write(f"*Current Goal*: {st.session_state.goal}")
        final_suggestion = get_final_goal_suggestion(st.session_state.goal)
        st.write(f"*Recommendation*: {final_suggestion}")
        st.markdown('</div>')

    # File upload
    uploaded_files = st.session_state.uploaded_files if st.session_state.uploaded_files else st.file_uploader("Upload your e-commerce CSV files", type=["csv"], accept_multiple_files=True)

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
                    filtered_df, suggestion_df, report_data, suggestion_data = process_file(uploaded_file, uploaded_file.name)
                    if filtered_df is not None:
                        all_dfs.append((uploaded_file.name, filtered_df))
                        all_suggestion_dfs.append((uploaded_file.name, suggestion_df))
                        # Add individual report to ZIP
                        zip_file.writestr(f"sentiment_report_{uploaded_file.name}.xlsx", report_data)
                        # Add suggestions report to ZIP
                        zip_file.writestr(f"top_suggestions_report_{uploaded_file.name}.xlsx", suggestion_data)

        # Display Goal Progress for Each File
        if st.session_state.goal != "None" and all_dfs:
            st.markdown('<div class="section-container">', unsafe_allow_html=True)
            st.subheader("Goal Progress Across Files")

            # Define goal-specific suggestions and explanations
            goal_suggestions = {
                "reduce shipping complaints by 20%": {
                    "suggestion": lambda neg_pct: f"Partner with reliable carriers to reduce shipping-related negative feedback, currently at {neg_pct:.1f}% for this file.",
                    "explanation": lambda neg_pct: (
                        f"With {neg_pct:.1f}% of reviews reflecting negative sentiment, many of which may relate to shipping issues, optimizing logistics is critical. "
                        "Partner with dependable carriers or offer real-time tracking to address complaints like delays or lost packages. "
                        "This aligns with your goal to reduce shipping complaints by 20%, enhancing customer trust and potentially converting negative feedback into positive experiences."
                    )
                },
                "increase positive product reviews by 15%": {
                    "suggestion": lambda pos_pct: f"Enhance product quality and descriptions to boost positive reviews, currently at {pos_pct:.1f}% for this file.",
                    "explanation": lambda pos_pct: (
                        f"Positive reviews constitute {pos_pct:.1f}% of feedback for this file, indicating a solid foundation to build upon. "
                        "Improving product quality through stricter quality checks and ensuring product descriptions set expectations accurately can further increase positive sentiment. "
                        "This strategy supports your goal to increase positive product reviews by 15%, boosting customer satisfaction and driving repeat purchases."
                    )
                },
                "convert neutral feedback to positive": {
                    "suggestion": lambda neu_pct: f"Engage neutral reviewers with follow-up offers to convert their feedback, currently at {neu_pct:.1f}% for this file, into positive.",
                    "explanation": lambda neu_pct: (
                        f"Neutral feedback, making up {neu_pct:.1f}% of reviews, represents an opportunity to shift sentiment toward positive. "
                        "Proactively reaching out to these customers with personalized offers, such as discounts or improved support, can enhance their experience. "
                        "This approach supports your goal to directly convert neutral feedback to positive, fostering stronger customer loyalty and improving overall sentiment."
                    )
                },
                "improve customer service responsiveness": {
                    "suggestion": lambda neg_pct, pos_pct: f"Train staff to improve responsiveness, addressing negative feedback at {neg_pct:.1f}% and boosting positive feedback at {pos_pct:.1f}% for this file.",
                    "explanation": lambda neg_pct, pos_pct: (
                        f"With negative feedback at {neg_pct:.1f}% and positive at {pos_pct:.1f}%, customer service responsiveness is a key area for improvement. "
                        "Training staff to handle inquiries with greater empathy and speed, or implementing 24/7 support channels, can reduce negative sentiment and enhance positive experiences. "
                        "This aligns with your goal to improve customer service responsiveness, driving higher satisfaction and better retention."
                    )
                },
                "enhance checkout process satisfaction": {
                    "suggestion": lambda neg_pct, pos_pct: f"Simplify the checkout process to reduce negative feedback at {neg_pct:.1f}% and increase positive feedback at {pos_pct:.1f}% for this file.",
                    "explanation": lambda neg_pct, pos_pct: (
                        f"Negative feedback at {neg_pct:.1f}% may include issues with the checkout process, while positive feedback at {pos_pct:.1f}% suggests some satisfaction. "
                        "Streamlining the checkout process by reducing steps, improving payment options, or enhancing website usability can address frustrations and elevate positive sentiment. "
                        "This supports your goal to enhance checkout process satisfaction, improving conversion rates and customer experience."
                    )
                },
                "custom": {
                    "suggestion": lambda pos_pct, neg_pct, neu_pct, goal: f"Align with your custom goal '{goal}' by addressing sentiment distribution: Positive {pos_pct:.1f}%, Negative {neg_pct:.1f}%, Neutral {neu_pct:.1f}% for this file.",
                    "explanation": lambda pos_pct, neg_pct, neu_pct, goal: (
                        f"Your custom goal '{goal}' can be supported by analyzing the sentiment distribution: {pos_pct:.1f}% positive, {neg_pct:.1f}% negative, and {neu_pct:.1f}% neutral. "
                        "Focus on reducing negative feedback through targeted improvements in key areas like product or service quality, while leveraging positive sentiment to reinforce strengths. "
                        "Engaging neutral consumers with follow-ups can also shift sentiment, aligning with your unique objectives to optimize e-commerce performance."
                    )
                }
            }

            for file_name, filtered_df in all_dfs:
                if 'Sentiment_VADER' not in filtered_df.columns or filtered_df.empty:
                    st.warning(f"No valid sentiment data available for {file_name}.")
                    continue
                sentiment_counts = filtered_df['Sentiment_VADER'].value_counts()
                total_reviews = len(filtered_df)
                if total_reviews == 0:
                    st.write(f"{file_name}: No reviews available to calculate sentiment distribution.")
                    continue
                negative_pct = (sentiment_counts.get('Negative', 0) / total_reviews * 100)
                positive_pct = (sentiment_counts.get('Positive', 0) / total_reviews * 100)
                neutral_pct = (sentiment_counts.get('Neutral', 0) / total_reviews * 100)

                # Get the appropriate suggestion based on the goal
                goal_key = "custom" if "custom" in st.session_state.goal.lower() else st.session_state.goal.lower()
                suggestion_info = goal_suggestions.get(goal_key, goal_suggestions["custom"])

                # Generate suggestion
                if goal_key == "custom":
                    suggestion = suggestion_info["suggestion"](positive_pct, negative_pct, neutral_pct, st.session_state.goal)
                    explanation = suggestion_info["explanation"](positive_pct, negative_pct, neutral_pct, st.session_state.goal)
                elif goal_key in ["improve customer service responsiveness", "enhance checkout process satisfaction"]:
                    suggestion = suggestion_info["suggestion"](negative_pct, positive_pct)
                    explanation = suggestion_info["explanation"](negative_pct, positive_pct)
                elif goal_key == "convert neutral feedback to positive":
                    suggestion = suggestion_info["suggestion"](neutral_pct)
                    explanation = suggestion_info["explanation"](neutral_pct)
                else:  # reduce shipping complaints, increase positive product reviews
                    suggestion = suggestion_info["suggestion"](negative_pct if "reduce" in goal_key else positive_pct)
                    explanation = suggestion_info["explanation"](negative_pct if "reduce" in goal_key else positive_pct)

                # Display suggestion
                st.write(f"{file_name}: Sentiment distribution - Positive: {positive_pct:.1f}%, Negative: {negative_pct:.1f}%, Neutral: {neutral_pct:.1f}%.")
                st.markdown(f"*Suggestion*: {suggestion}")
                st.markdown(f"*Why This Matters*: {explanation}")
                st.markdown("---")
            
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

        # End Process button
        st.markdown('<div class="end-process-container">', unsafe_allow_html=True)
        if st.button("End Process and Start Over", key="end_process"):
            reset_app()
        st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="footer">
        © 2025 Sentiment Insights | Copyright S Kajamalan @Detech
    </div>
    """, unsafe_allow_html=True)