from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import os
import io
import base64
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

print("Checking NLTK data...")
try:
    nltk.data.find('tokenizers/punkt_tab')
    print("NLTK data already available.")
except (LookupError, OSError):
    print("Downloading required NLTK data...")
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('omw-1.4')
    print("NLTK data download complete!")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

global_data = {
    'df': None,
    'models': {},
    'vectorizer': None,
    'results': {},
    'preprocessor': None,
    'X_train': None,
    'X_test': None,
    'y_train': None,
    'y_test': None
}

class TextPreprocessor:
    """Text preprocessing for sentiment analysis."""

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        negations = {'no', 'not', 'nor', 'never', 'neither', 'nobody', 'nothing', 'nowhere'}
        self.stop_words = self.stop_words - negations

    def clean_text(self, text):
        """Clean and normalize text."""
        if pd.isna(text) or text == '':
            return ''

        text = text.lower()
        text = re.sub(r'<.*?>', '', text)

        contractions = {
            "won't": "will not", "can't": "cannot", "n't": " not",
            "'re": " are", "'ve": " have", "'ll": " will",
            "'d": " would", "'m": " am"
        }
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)

        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def tokenize_and_lemmatize(self, text):
        """Tokenize and lemmatize text."""
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens
                  if token not in self.stop_words and len(token) > 2]
        return tokens

    def preprocess(self, text):
        """Complete preprocessing pipeline."""
        cleaned = self.clean_text(text)
        tokens = self.tokenize_and_lemmatize(cleaned)
        return ' '.join(tokens)

    def add_features(self, df):
        """Engineer additional features."""
        df = df.copy()
        df['text_length'] = df['review_text'].apply(lambda x: len(str(x)))
        df['word_count'] = df['review_text'].apply(lambda x: len(str(x).split()))
        df['polarity'] = df['review_text'].apply(
            lambda x: TextBlob(str(x)).sentiment.polarity
        )
        df['subjectivity'] = df['review_text'].apply(
            lambda x: TextBlob(str(x)).sentiment.subjectivity
        )
        return df

def plot_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64

def train_models(X_train, y_train):
    models = {
        'Naive Bayes': MultinomialNB(alpha=1.0),
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'SVM': SVC(
            kernel='rbf',
            class_weight='balanced',
            probability=True,
            random_state=42
        )
    }

    for name, model in models.items():
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv,
                                   scoring='f1', n_jobs=-1)
        model.fit(X_train, y_train)
        models[name] = model

    return models

def evaluate_models(models, X_test, y_test):
    results = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='binary'),
            'recall': recall_score(y_test, y_pred, average='binary'),
            'f1_score': f1_score(y_test, y_pred, average='binary'),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }

    return results

@app.route('/')
def index():
    csv_path = 'amazon_reviews_prepared.csv'
    file_exists = os.path.exists(csv_path)

    initial_stats = None
    auto_loaded = False

    if file_exists:
        try:
            df = pd.read_csv(csv_path)

            required_cols = ['review_text', 'rating']
            if all(col in df.columns for col in required_cols):
                preprocessor = TextPreprocessor()

                df['cleaned_text'] = df['review_text'].apply(preprocessor.clean_text)
                df['processed_text'] = df['review_text'].apply(preprocessor.preprocess)
                df = preprocessor.add_features(df)

                df = df[df['rating'] != 3].copy()
                df['sentiment'] = (df['rating'] >= 4).astype(int)

                initial_stats = {
                    'total_reviews': len(df),
                    'positive_reviews': int((df['sentiment'] == 1).sum()),
                    'negative_reviews': int((df['sentiment'] == 0).sum()),
                    'avg_word_count': float(df['word_count'].mean()),
                    'avg_polarity': float(df['polarity'].mean()),
                    'rating_distribution': df['rating'].value_counts().sort_index().to_dict()
                }
                auto_loaded = True

                global_data['df'] = df
                global_data['preprocessor'] = preprocessor
        except Exception as e:
            print(f"Error auto-loading CSV: {e}")

    return render_template('index.html',
                           file_exists=file_exists,
                           auto_loaded=auto_loaded,
                           initial_stats=initial_stats)

@app.route('/auto-load', methods=['GET'])
def auto_load():
    try:
        csv_path = 'amazon_reviews_prepared.csv'

        if not os.path.exists(csv_path):
            return jsonify({
                'success': False,
                'error': f'File not found: {csv_path}. Please ensure amazon_reviews_prepared.csv is in the same directory as the Flask app.'
            }), 404

        df = pd.read_csv(csv_path)

        required_cols = ['review_text', 'rating']
        if not all(col in df.columns for col in required_cols):
            return jsonify({
                'success': False,
                'error': f'CSV must contain columns: {required_cols}. Found: {list(df.columns)}'
            }), 400

        global_data['preprocessor'] = TextPreprocessor()

        df['cleaned_text'] = df['review_text'].apply(global_data['preprocessor'].clean_text)
        df['processed_text'] = df['review_text'].apply(global_data['preprocessor'].preprocess)
        df = global_data['preprocessor'].add_features(df)

        df = df[df['rating'] != 3].copy()

        if len(df) < 10:
            return jsonify({
                'success': False,
                'error': 'Not enough data. Need at least 10 non-neutral reviews (excluding 3-star ratings).'
            }), 400

        df['sentiment'] = (df['rating'] >= 4).astype(int)

        positive_count = (df['sentiment'] == 1).sum()
        negative_count = (df['sentiment'] == 0).sum()

        if positive_count < 2 or negative_count < 2:
            return jsonify({
                'success': False,
                'error': f'Insufficient class balance. Need at least 2 reviews in each sentiment class. Currently: {positive_count} positive, {negative_count} negative.'
            }), 400

        global_data['df'] = df

        stats = {
            'total_reviews': len(df),
            'positive_reviews': int((df['sentiment'] == 1).sum()),
            'negative_reviews': int((df['sentiment'] == 0).sum()),
            'avg_word_count': float(df['word_count'].mean()),
            'avg_polarity': float(df['polarity'].mean()),
            'rating_distribution': df['rating'].value_counts().sort_index().to_dict()
        }

        return jsonify({
            'success': True,
            'message': f'Successfully loaded {len(df)} reviews from {csv_path}!',
            'stats': stats
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/train', methods=['POST'])
def train():
    try:
        if global_data['df'] is None:
            return jsonify({'error': 'Please load data first'}), 400

        df = global_data['df']

        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )

        X = vectorizer.fit_transform(df['processed_text'])
        y = df['sentiment']

        total_samples = len(df)
        if total_samples < 10:
            return jsonify({'error': 'Dataset too small. Need at least 10 reviews.'}), 400

        if total_samples < 50:
            test_size = 0.3
            min_test_samples = max(2, int(total_samples * 0.3))
        else:
            test_size = 0.2
            min_test_samples = max(4, int(total_samples * 0.2))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        global_data['vectorizer'] = vectorizer
        global_data['X_train'] = X_train
        global_data['X_test'] = X_test
        global_data['y_train'] = y_train
        global_data['y_test'] = y_test

        models = train_models(X_train, y_train)
        global_data['models'] = models

        results = evaluate_models(models, X_test, y_test)
        global_data['results'] = results

        results_json = {}
        for name, metrics in results.items():
            results_json[name] = {
                'accuracy': float(metrics['accuracy']),
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'f1_score': float(metrics['f1_score'])
            }

        return jsonify({
            'success': True,
            'message': 'Models trained successfully!',
            'results': results_json
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not global_data['models']:
            return jsonify({'error': 'Please train models first'}), 400

        data = request.get_json()
        review_text = data.get('review_text', '')

        if not review_text:
            return jsonify({'error': 'No review text provided'}), 400

        processed_text = global_data['preprocessor'].preprocess(review_text)

        X = global_data['vectorizer'].transform([processed_text])

        predictions = {}
        for name, model in global_data['models'].items():
            pred = model.predict(X)[0]
            pred_proba = model.predict_proba(X)[0] if hasattr(model, 'predict_proba') else None

            predictions[name] = {
                'sentiment': 'Positive' if pred == 1 else 'Negative',
                'confidence': float(pred_proba[pred]) if pred_proba is not None else None
            }

        return jsonify({
            'success': True,
            'predictions': predictions,
            'processed_text': processed_text
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/visualizations')
def visualizations():
    try:
        if global_data['df'] is None or not global_data['results']:
            return jsonify({'error': 'Please load data and train models first'}), 400

        df = global_data['df']
        results = global_data['results']
        y_test = global_data['y_test']

        plots = {}

        fig, ax = plt.subplots(figsize=(10, 6))
        rating_counts = df['rating'].value_counts().sort_index()
        ax.bar(rating_counts.index, rating_counts.values, color='skyblue', edgecolor='black')
        ax.set_xlabel('Rating (Stars)', fontsize=12)
        ax.set_ylabel('Number of Reviews', fontsize=12)
        ax.set_title('Distribution of Review Ratings', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plots['rating_dist'] = plot_to_base64(fig)

        fig, ax = plt.subplots(figsize=(12, 6))
        models = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        x = np.arange(len(models))
        width = 0.2

        for i, metric in enumerate(metrics):
            values = [results[model][metric] for model in models]
            ax.bar(x + i*width, values, width, label=metric.replace('_', ' ').title())

        ax.set_xlabel('Models', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plots['model_comparison'] = plot_to_base64(fig)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()

        for idx, (name, result) in enumerate(results.items()):
            cm = result['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(f'{name}\nF1-Score: {result["f1_score"]:.4f}')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')

        plt.tight_layout()
        plots['confusion_matrices'] = plot_to_base64(fig)

        fig, ax = plt.subplots(figsize=(10, 8))

        for name, result in results.items():
            if result['y_pred_proba'] is not None:
                fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
                auc = roc_auc_score(y_test, result['y_pred_proba'])
                ax.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)

        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves - Model Comparison')
        ax.legend()
        ax.grid(alpha=0.3)
        plots['roc_curves'] = plot_to_base64(fig)

        positive_reviews = df[df['sentiment'] == 1]['processed_text'].tolist()
        negative_reviews = df[df['sentiment'] == 0]['processed_text'].tolist()

        if positive_reviews:
            fig, ax = plt.subplots(figsize=(12, 6))
            wordcloud = WordCloud(width=1200, height=600, background_color='white',
                                 colormap='Greens', max_words=100).generate(' '.join(positive_reviews))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title('Positive Reviews Word Cloud', fontsize=16, fontweight='bold')
            plots['wordcloud_positive'] = plot_to_base64(fig)

        if negative_reviews:
            fig, ax = plt.subplots(figsize=(12, 6))
            wordcloud = WordCloud(width=1200, height=600, background_color='white',
                                 colormap='Reds', max_words=100).generate(' '.join(negative_reviews))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title('Negative Reviews Word Cloud', fontsize=16, fontweight='bold')
            plots['wordcloud_negative'] = plot_to_base64(fig)

        return jsonify({
            'success': True,
            'plots': plots
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/feature-importance')
def feature_importance():
    try:
        if 'Logistic Regression' not in global_data['models']:
            return jsonify({'error': 'Please train models first'}), 400

        model = global_data['models']['Logistic Regression']
        vectorizer = global_data['vectorizer']

        feature_names = vectorizer.get_feature_names_out()
        coefficients = model.coef_[0]

        top_positive_idx = np.argsort(coefficients)[-20:]
        positive_features = [
            {'word': feature_names[idx], 'importance': float(coefficients[idx])}
            for idx in reversed(top_positive_idx)
        ]

        top_negative_idx = np.argsort(coefficients)[:20]
        negative_features = [
            {'word': feature_names[idx], 'importance': float(coefficients[idx])}
            for idx in top_negative_idx
        ]

        return jsonify({
            'success': True,
            'positive_features': positive_features,
            'negative_features': negative_features
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/check-file')
def check_file():
    csv_path = 'amazon_reviews_prepared.csv'
    file_exists = os.path.exists(csv_path)

    return jsonify({
        'file_exists': file_exists,
        'file_path': os.path.abspath(csv_path) if file_exists else None
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)