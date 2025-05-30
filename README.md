# Movie Review Sentiment Analysis

A machine learning project that analyzes the sentiment of movie reviews using Natural Language Processing (NLP) techniques and the Naive Bayes classifier.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Environment Setup](#environment-setup)
- [Data Source](#data-source)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Visualizations](#visualizations)
- [Credits](#credits)

## Overview

This project performs sentiment analysis on movie reviews to classify them as either positive or negative. The system uses text preprocessing techniques, TF-IDF vectorization, and a Multinomial Naive Bayes classifier to achieve accurate sentiment predictions.

## Features

- **Text Preprocessing**: HTML tag removal, text normalization, tokenization, stop word removal, and lemmatization
- **Feature Extraction**: TF-IDF (Term Frequency-Inverse Document Frequency) vectorization
- **Machine Learning**: Multinomial Naive Bayes classifier for sentiment prediction
- **Data Visualization**: 
  - Sentiment distribution plots
  - Confusion matrix heatmap
  - Word clouds for positive and negative reviews
- **Interactive Prediction**: Real-time sentiment prediction for user input

## Environment Setup

### Prerequisites
- Python 3.12 or higher
- pip package manager

### Required Libraries
```
pandas
numpy
scikit-learn
nltk
matplotlib
seaborn
wordcloud
re (built-in)
```

## Data Source

This project uses the **IMDB Dataset of 50K Movie Reviews** from Kaggle, which contains 50,000 movie reviews labeled as positive or negative sentiment.

**Dataset Details:**
- **Source**: [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Size**: 50,000 reviews
- **Format**: CSV file with 'review' and 'sentiment' columns
- **Split**: Balanced dataset with equal positive and negative reviews

## Installation

1. **Clone or download the project files**

2. **Install required packages**:
   ```bash
   pip install pandas numpy scikit-learn nltk matplotlib seaborn wordcloud
   ```

3. **Download the dataset**:
   - Visit the [Kaggle dataset page](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
   - Download the `IMDB Dataset.csv` file
   - Rename it to `imdb.csv` and place it in your project directory

4. **Download NLTK data** (this will be done automatically when you run the code):
   - punkt
   - stopwords
   - wordnet
   - punkt_tab

## How to Run

1. **Ensure you have the dataset file**:
   Make sure `imdb.csv` is in the same directory as your Python script.

2. **Run the complete analysis**:
   ```bash
   python movie_review_py_version.py
   ```

3. **What the script does**:
   - Loads and explores the dataset
   - Preprocesses the text data (cleaning, tokenization, lemmatization)
   - Splits data into training and testing sets (80/20 split)
   - Trains a Multinomial Naive Bayes model using TF-IDF features
   - Evaluates model performance with accuracy score and classification report
   - Generates visualizations:
     - Sentiment distribution plot
     - Confusion matrix
     - Word clouds for positive and negative reviews
   - Allows interactive sentiment prediction for user input

4. **Interactive prediction**:
   When prompted, enter a movie review to get real-time sentiment prediction.

## Project Structure

```
movie-sentiment-analysis/
│
├── imdb.csv                    # Dataset file (download required)
├── movie_review_py_version.py  # Main Python script
├── movie_review.ipynb          # Jupyter file version
├── README.md                   # This file
└── requirements.txt            # Dependencies list
```

## Model Performance

The Naive Bayes classifier typically achieves:
- **Accuracy**: ~85-87% on the test set
- **Features**: 5,000 most important TF-IDF features
- **Training Time**: Fast training and prediction

Performance metrics include:
- Accuracy score
- Precision, Recall, and F1-score for both classes
- Confusion matrix visualization

## Visualizations

The project generates several informative visualizations:

1. **Sentiment Distribution**: Bar chart showing the balance between positive and negative reviews
2. **Confusion Matrix**: Heatmap displaying model prediction accuracy
3. **Word Clouds**: 
   - Most frequent words in positive reviews
   - Most frequent words in negative reviews

## Credits

### Dataset
This project uses the **IMDB Dataset of 50K Movie Reviews** created by Lakshmi Narayana Pathi.

- **Dataset Source**: [Kaggle - IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Original Data**: Internet Movie Database (IMDB)
- **License**: Please refer to the Kaggle dataset page for licensing information

### Acknowledgments
- IMDB for providing the original movie review data
- Kaggle community for dataset maintenance and accessibility
- NLTK team for natural language processing tools
- Scikit-learn team for machine learning utilities

## Developers

This project was developed by:

- **Banzuela, Allan Jr D.** - [@Banzuela1319](https://github.com/Banzuela1319)
- **Furaque, John Patrick F.** - [@furaquejohn11](https://github.com/furaquejohn11)
- **Valdesco, Apple S.** - [@itsapple15](https://github.com/itsapple15)
---

**Note**: This project is for educational and research purposes. Please ensure you comply with the dataset's terms of use and licensing requirements.