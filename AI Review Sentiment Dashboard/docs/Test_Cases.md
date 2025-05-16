1. Review Upload Functionality

Test Case 1.1: Upload Valid Review File

     Description: Verify that users can upload a valid review file (CSV, TXT) with correct format.

Steps:
   1. Navigate to the "Upload Reviews" section.
   2. Click "Choose File" and select a valid CSV file.
   3. Click "Upload".

Expected Result: File uploads successfully, success message displayed, reviews appear in the dashboard.

Test Data: CSV with 1000 reviews


Test Case 1.2: Upload Empty File

     Description: Verify system handles empty review files.

Steps:
   1. Navigate to "Upload Reviews".
   2. Select empty CSV file.
   3. Click "Upload".

Expected Result: Error message: "File is empty. Please upload a valid file with reviews."

Test Data: empty.csv.

2. Sentiment Analysis Functionality

Test Case 2.1: Analyze Positive Reviews

     Description: Verify sentiment analysis correctly identifies positive reviews using VADER, TextBlob, or BERT.

Steps:
   1. Upload CSV with positive reviews 
   2. Trigger sentiment analysis.
   3. View results in dashboard.

Expected Result: Reviews tagged as "Positive" with high sentiment scores (e.g., VADER compound score > 0.5).

Test Data: "Amazing service!", "Really happy with the food."

Test Case 2.2: Analyze Mixed Sentiment Reviews

     Description: Verify system handles reviews with mixed sentiments.

Steps:
   1. Upload CSV with mixed reviews (e.g., "Food was great, but service was slow.").
   2. Trigger sentiment analysis.

Expected Result: Reviews tagged as "Neutral" or split into positive/negative components, depending on model (BERT splits better).

Test Data: "Food was great, but service was slow."

Test Case 2.3: Handle Non-English Reviews

     Description: Verify system flags or skips non-English reviews (unless multilingual BERT is used).

Steps:
   1. Upload CSV with reviews in Spanish (e.g., "¡Excelente producto!").
   2. Trigger sentiment analysis.

Expected Result: Non-English reviews flagged with warning: "Non-English review detected, analysis skipped" (or processed if multilingual BERT is enabled).

Test Data: "¡Excelente producto!", "Produit incroyable!"

3. Tag Cloud Generation

Test Case 3.1: Generate Tag Cloud for Reviews

     Description: Verify tag cloud is generated based on frequent words in reviews.

Steps:
   1. Navigate to "Tag Cloud" section.
   2. Click "Generate Tag Cloud".

Expected Result: Tag cloud displays with words like "great", "service", "product" sized by frequency.

Test Data: 50 reviews with repeated words like "great", "fast", "poor".


4. Dashboard Visualization

Test Case 4.1: Display Sentiment Trends

     Description: Verify sentiment trends are displayed correctly over time.

Steps:
   1. Navigate to "Sentiment Trends" section.
   2. Select date range 

Expected Result: bar chart shows positive, negative, neutral trends over time.

Test Data: CSV with reviews 

Test Case 4.2: Dashboard Access for Tiered Plans

     Description: Verify access restrictions based on pricing tier (review volume limits).

Steps:
   1. Upload CSV with 1000 reviews.
   2. Attempt to process all reviews.

Expected Result: Warning: "Review limit exceeded for your plan. Upgrade to process more."

Test Data: CSV with 1000 reviews.

5. Performance and Scalability

Test Case 5.1: Handle Large Review Volume

     Description: Verify system processes large review volumes within acceptable time.

Steps:
   1. Upload CSV with 1000 reviews.
   2. Trigger sentiment analysis and tag cloud generation.

Expected Result: Processing completes within 5 minutes, results displayed correctly.

Test Data: CSV with 1000 reviews.

