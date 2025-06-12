1. Functional Requirements
     Users can upload reviews in CSV, TXT, or JSON format
     System classifies reviews as Positive, Neutral, or Negative
     Users can choose the NLP engine: Vader, TextBlob, or BERT
     System shows trend graphs and word clouds
     Users can export reports as CSV or PDF
     Pricing depends on the number of reviews processed

2. Non-Functional Requirements
     Dashboard responds within 3 seconds for up to 10,000 reviews
     Can be deployed on Streamlit Cloud or similar platforms
     Code is easy to maintain and well documented
     User interface is fast and works well on mobile and desktop

3. Constraints
     Use only free or open-source tools at first
     User data is not saved permanently unless exported
     Models run locally or via free Hugging Face API

4. Assumptions
     Reviews have enough info to detect sentiment
     Reviews are mostly in English

5. Dependencies
     Python libraries: pandas, nltk, textblob, transformers, wordcloud, matplotlib, plotly
     Streamlit for frontend
     Hugging Face API for BERT model