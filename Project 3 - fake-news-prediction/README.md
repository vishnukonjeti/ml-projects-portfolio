# Fake News Prediction

This project builds an end-to-end Natural Language Processing (NLP) and Machine Learning classification system in Python to detect whether a news article is **Real** (reliable) or **Fake** (unreliable) based on its title and author.

---

## 📌 Project Overview

- **Domain**: Natural Language Processing (NLP) / Information Credibility / Text Classification
- **Problem Type**: Supervised Binary Classification
- **Algorithm**: Logistic Regression with TF-IDF Vectorization
- **Primary Goal**: Classify news articles as Reliable (`0`) or Unreliable / Fake (`1`) using textual features.

---

## 📊 Dataset Details

The dataset used is the standard **Fake News Dataset** (containing labeled news articles):

- **File**: `fake_news_dataset.csv`
- **Total Records**: 20,800 news articles
- **Columns**:
  - `id`: Unique identifier for each news article
  - `title`: The headline / title of the article (558 missing values handled)
  - `author`: The author of the news article (1,957 missing values handled)
  - `text`: The body text of the article (39 missing values handled)
  - `label`: Target indicator:
    - `0` : Real / Reliable news
    - `1` : Fake / Unreliable news

---

## ⚙️ Machine Learning Pipeline

1. **Data Ingestion & Missing Value Handling**:
   - Loaded into a Pandas DataFrame.
   - Evaluated data shape `(20800, 5)` and missing value counts across columns.
   - Imputed null/missing values with empty strings (`''`) to ensure clean text concatenation.

2. **Feature Engineering**:
   - Combined the `author` and `title` attributes into a unified text field:
     $$\text{content} = \text{author} + \text{" "} + \text{title}$$

3. **Text Preprocessing & NLP Pipeline**:
   - **Regular Expression Filtering**: Replaced non-alphabetic characters (`[^a-zA-Z]`) with spaces to eliminate punctuation, numbers, and special symbols.
   - **Case Normalization**: Converted all text to lowercase.
   - **Tokenization**: Split text strings into individual word tokens.
   - **Stopword Removal**: Filtered out common English stopwords using NLTK's `stopwords.words('english')`.
   - **Stemming**: Applied NLTK's `PorterStemmer` to reduce words to their base/root forms (e.g., *playing*, *plays* $\rightarrow$ *play*).

4. **Feature Extraction (TF-IDF Vectorization)**:
   - Converted preprocessed textual data into numerical feature matrices using `TfidfVectorizer()`.
   - Produced a sparse matrix representation consisting of 20,800 instances across 17,128 unique vocabulary features based on Term Frequency - Inverse Document Frequency.

5. **Stratified Train-Test Split**:
   - Split ratio: 80% training (16,640 samples) and 20% testing (4,160 samples).
   - Stratified sampling (`stratify=y`) to maintain class proportions across both splits.
   - `random_state = 2` for deterministic reproducibility.

6. **Model Training**:
   - Trained a `LogisticRegression` classifier on the TF-IDF feature matrix.

7. **Model Evaluation**:
   - **Training Accuracy**: ~98.64%
   - **Testing Accuracy**: ~97.91%

8. **Predictive System**:
   - Implemented an inference workflow that evaluates news samples:
     - Preprocesses and vectorizes the text content using the fitted TF-IDF vocabulary.
     - Performs classification with the trained Logistic Regression model.
     - Interprets the output: `0` $\rightarrow$ *"The news is real"*, `1` $\rightarrow$ *"The news is fake"*.

---

## 📁 Project Structure

```text
Project 3 - fake-news-prediction/
├── fake-news-prediction.ipynb   # Main Jupyter notebook with NLP pipeline, TF-IDF, and model
├── fake_news_dataset.csv        # Labeled fake news dataset (20,800 articles)
└── README.md                    # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.9+ installed along with dependencies from the repository's root:

```bash
pip install -r ../requirements.txt
```

### 2. Download NLTK Stopwords
The notebook automatically downloads the required stopwords data:
```python
import nltk
nltk.download('stopwords')
```

### 3. Launch the Notebook
Open the project directory in VS Code or run Jupyter directly:

```bash
jupyter notebook "fake-news-prediction.ipynb"
```

Run all cells to execute text preprocessing, TF-IDF vectorization, Logistic Regression training, accuracy evaluation, and sample predictions.
