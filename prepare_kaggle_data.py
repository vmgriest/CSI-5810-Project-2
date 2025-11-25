"""
Data Preparation Script for Kaggle Amazon Reviews Dataset
CSI 5810 Project 2

This script downloads and prepares the Amazon product reviews dataset from Kaggle.
"""

import pandas as pd
import os
import kagglehub

def download_and_prepare_data():
    """Download Kaggle dataset and prepare it for the Flask app."""
    
    print("="*70)
    print("DOWNLOADING KAGGLE DATASET")
    print("="*70)
    
    # Download latest version
    path = kagglehub.dataset_download("datafiniti/consumer-reviews-of-amazon-products")
    print(f"\nPath to dataset files: {path}")
    
    # Find CSV files in the downloaded path
    csv_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    print(f"\nFound {len(csv_files)} CSV file(s):")
    for f in csv_files:
        print(f"  - {f}")
    
    if not csv_files:
        print("\nError: No CSV files found in the dataset!")
        return None
    
    # Load the main dataset
    print(f"\nLoading dataset from: {csv_files[0]}")
    df = pd.read_csv(csv_files[0])
    
    print(f"\nOriginal dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Display first few rows
    print("\nFirst 3 rows:")
    print(df.head(3))
    
    # Identify relevant columns
    # Common column names in Amazon reviews datasets
    possible_text_cols = ['reviews.text', 'reviewText', 'review_text', 'text', 'review']
    possible_rating_cols = ['reviews.rating', 'rating', 'overall', 'reviews.overall']
    possible_date_cols = ['reviews.date', 'reviewTime', 'date', 'review_date']
    
    # Find the actual column names
    text_col = None
    rating_col = None
    date_col = None
    
    for col in possible_text_cols:
        if col in df.columns:
            text_col = col
            break
    
    for col in possible_rating_cols:
        if col in df.columns:
            rating_col = col
            break
            
    for col in possible_date_cols:
        if col in df.columns:
            date_col = col
            break
    
    if not text_col or not rating_col:
        print("\nError: Could not find required columns!")
        print(f"Looking for text column in: {possible_text_cols}")
        print(f"Looking for rating column in: {possible_rating_cols}")
        return None
    
    print(f"\nIdentified columns:")
    print(f"  - Text column: {text_col}")
    print(f"  - Rating column: {rating_col}")
    print(f"  - Date column: {date_col if date_col else 'Not found'}")
    
    # Create standardized dataset
    prepared_df = pd.DataFrame()
    prepared_df['review_text'] = df[text_col]
    prepared_df['rating'] = df[rating_col]
    
    if date_col:
        prepared_df['date'] = df[date_col]
    else:
        prepared_df['date'] = '2025-01-01'
    
    # Add additional useful columns if available
    if 'name' in df.columns:
        prepared_df['product_name'] = df['name']
    elif 'reviews.title' in df.columns:
        prepared_df['product_name'] = df['reviews.title']
    
    if 'reviews.doRecommend' in df.columns:
        prepared_df['recommend'] = df['reviews.doRecommend']
    
    # Clean the data
    print("\n" + "="*70)
    print("CLEANING DATA")
    print("="*70)
    
    initial_count = len(prepared_df)
    print(f"Initial rows: {initial_count}")
    
    # Remove rows with missing text or rating
    prepared_df = prepared_df.dropna(subset=['review_text', 'rating'])
    print(f"After removing missing values: {len(prepared_df)} rows")
    
    # Remove empty reviews
    prepared_df = prepared_df[prepared_df['review_text'].str.strip() != '']
    print(f"After removing empty reviews: {len(prepared_df)} rows")
    
    # Convert rating to numeric and filter valid ratings (1-5)
    prepared_df['rating'] = pd.to_numeric(prepared_df['rating'], errors='coerce')
    prepared_df = prepared_df[prepared_df['rating'].between(1, 5)]
    print(f"After filtering valid ratings (1-5): {len(prepared_df)} rows")
    
    # Remove very short reviews (less than 10 characters)
    prepared_df = prepared_df[prepared_df['review_text'].str.len() >= 10]
    print(f"After removing very short reviews: {len(prepared_df)} rows")
    
    # Sample data for manageability (optional - adjust as needed)
    if len(prepared_df) > 10000:
        print(f"\nDataset is large ({len(prepared_df)} rows). Sampling 10,000 reviews...")
        prepared_df = prepared_df.sample(n=10000, random_state=42)
    
    # Display statistics
    print("\n" + "="*70)
    print("DATASET STATISTICS")
    print("="*70)
    print(f"Total reviews: {len(prepared_df)}")
    print(f"\nRating distribution:")
    print(prepared_df['rating'].value_counts().sort_index())
    print(f"\nAverage review length: {prepared_df['review_text'].str.len().mean():.1f} characters")
    print(f"Average words per review: {prepared_df['review_text'].str.split().str.len().mean():.1f} words")
    
    # Check sentiment balance
    positive_reviews = len(prepared_df[prepared_df['rating'] >= 4])
    negative_reviews = len(prepared_df[prepared_df['rating'] <= 2])
    neutral_reviews = len(prepared_df[prepared_df['rating'] == 3])
    
    print(f"\nSentiment distribution:")
    print(f"  Positive (4-5 stars): {positive_reviews} ({positive_reviews/len(prepared_df)*100:.1f}%)")
    print(f"  Neutral (3 stars): {neutral_reviews} ({neutral_reviews/len(prepared_df)*100:.1f}%)")
    print(f"  Negative (1-2 stars): {negative_reviews} ({negative_reviews/len(prepared_df)*100:.1f}%)")
    
    # Save to CSV
    output_file = 'amazon_reviews_prepared.csv'
    prepared_df.to_csv(output_file, index=False)
    print(f"\n" + "="*70)
    print(f"SUCCESS! Data saved to: {output_file}")
    print("="*70)
    print("\nYou can now upload this file to the Flask application!")
    
    # Display sample reviews
    print("\n" + "="*70)
    print("SAMPLE REVIEWS")
    print("="*70)
    
    print("\n✅ Sample Positive Review (5-star):")
    positive_sample = prepared_df[prepared_df['rating'] == 5].sample(1)
    print(positive_sample['review_text'].values[0][:200] + "...")
    
    print("\n❌ Sample Negative Review (1-star):")
    negative_sample = prepared_df[prepared_df['rating'] == 1].sample(1)
    print(negative_sample['review_text'].values[0][:200] + "...")
    
    return prepared_df

if __name__ == "__main__":
    try:
        df = download_and_prepare_data()
        if df is not None:
            print("\n✨ Data preparation complete!")
            print("Next steps:")
            print("  1. Run the Flask app: python app.py")
            print("  2. Open http://localhost:5000 in your browser")
            print("  3. Upload the 'amazon_reviews_prepared.csv' file")
            print("  4. Train models and analyze sentiment!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have kagglehub installed:")
        print("  pip install kagglehub")