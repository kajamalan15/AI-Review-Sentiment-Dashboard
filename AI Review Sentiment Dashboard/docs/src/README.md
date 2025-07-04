# AI-Review-Sentiment-Dashboard

Project Overview :-

The AI Review Sentiment Dashboard is a web-based application designed for e-commerce businesses, restaurants, and service providers to analyze customer reviews. It allows users to upload review files, visualize sentiment trends, and generate tag clouds to identify key themes. The application uses natural language processing (NLP) tools including VADER, TextBlob, and BERT (via Hugging Face free tier) for sentiment analysis. The dashboard operates on a tiered pricing model based on review volume, with a hosted solution for easy access.

    Frontend: Streamlit for an interactive, user-friendly interface.
    Backend: Python for processing reviews, performing sentiment analysis, and generating visualizations.
    Target Clients: E-commerce, restaurants, service providers.
    Features:
         Upload reviews in CSV or TXT format.
         Display sentiment trends (positive, negative, neutral) over time.
         Generate tag clouds for frequent words in reviews.
         
Prerequisites :-

  Python 3.12.6
  Streamlit
  NLP libraries: vaderSentiment, textblob, transformers (Hugging Face)
  Additional dependencies: pandas, numpy, matplotlib, wordcloud
  Internet access for Hugging Face BERT model (free tier)


streamlit==1.38.0
pandas==2.2.3
matplotlib==3.10.1
plotly==6.1.0
wordcloud==1.9.4
vaderSentiment==3.3.2
textblob==0.19.0
transformers==4.51.3
chardet==3.0.4
openpyxl==3.1.5
torch==2.6.0

Run :- streamlit run app.py
