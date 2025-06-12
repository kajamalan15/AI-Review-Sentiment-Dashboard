# 03_SRS.md
# Software Requirements Specification (SRS)
# AI Review Sentiment Dashboard (Local NLP)

1. Introduction
Purpose:
   To describe the main features and requirements of the AI Review Sentiment Dashboard.

Scope:
   A web dashboard that lets users upload reviews, analyzes their sentiment, and shows easy-to-understand visuals using Streamlit.

Definitions:
   NLP: Natural Language Processing
   BERT: A language model for understanding text

2. Overall Description
Product Overview:
   A Python and Streamlit app that analyzes reviews locally using open-source tools and APIs.

Main Features:
   Upload reviews on the website
   Analyze sentiment with Vader, TextBlob, or BERT
   Show results with charts and word clouds
   Download reports as PDF or CSV

Users:
    Business users: Non-technical, want simple and clear info
    Developers: Can update or improve the app

Environment:
    Works on Windows, macOS, Linux
    Supported browsers: Chrome, Firefox, Safari
    Hosted on Streamlit Community Cloud

Constraints:
    Streamlit limits some design options
    BERT API calls are limited by free-tier quotas