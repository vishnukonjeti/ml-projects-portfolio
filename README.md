# Machine Learning Projects Portfolio

Welcome to my Machine Learning Projects Portfolio! This repository contains end-to-end machine learning projects, practical predictive systems, exploratory data analysis (EDA), and model experiments built using Python and Scikit-Learn.

---

## 📚 Portfolio Projects

| # | Project Name | Problem Type | Algorithm / Technique | Key Metrics | Link |
|---|--------------|--------------|-----------------------|-------------|------|
| 1 | **SONAR Rock vs. Mine Prediction** | Binary Classification | Logistic Regression, Stratified Sampling | **Train Acc**: ~83.4%<br>**Test Acc**: ~76.2% | [View Project](Project%201%20-%20sonar-rock-vs-mine-prediction/) |
| 2 | **Diabetes Prediction** | Binary Classification | Support Vector Machine (Linear SVM), Standardization | **Train Acc**: ~78.7%<br>**Test Acc**: ~77.3% | [View Project](Project%202%20-%20diabetes-prediction/) |
| 3 | **Fake News Prediction** | Binary Classification (NLP) | Logistic Regression, TF-IDF Vectorization, Porter Stemmer | **Train Acc**: ~98.6%<br>**Test Acc**: ~97.9% | [View Project](Project%203%20-%20fake-news-prediction/) |

---

## 📂 Repository Structure

```text
ml-practice-portfolio/
├── Project 1 - sonar-rock-vs-mine-prediction/
│   ├── Rock vs Mine Prediction.ipynb   # Exploratory analysis, training, and inference
│   ├── sonar_data.csv                  # Sonar signal dataset (60 features)
│   └── README.md                       # Project-specific documentation & workflow
├── Project 2 - diabetes-prediction/
│   ├── diabetes-prediction.ipynb       # Data exploration, scaling, SVM training, and inference
│   ├── diabetes.csv                    # Pima Indians diabetes dataset (8 features)
│   └── README.md                       # Project-specific documentation & workflow
├── Project 3 - fake-news-prediction/
│   ├── fake-news-prediction.ipynb      # NLP preprocessing, TF-IDF, Logistic Regression, inference
│   ├── fake_news_dataset.csv           # Kaggle fake news dataset (20,800 articles)
│   └── README.md                       # Project-specific documentation & workflow
├── .gitignore                          # Git ignore rules for Python, Jupyter, OS, and envs
├── LICENSE                             # MIT License
├── README.md                           # Portfolio overview and directory
└── requirements.txt                    # Shared project dependencies
```

---

## 🛠️ Tech Stack & Libraries

- **Language**: Python 3.9+
- **Data Manipulation & Analysis**: [NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/)
- **Natural Language Processing (NLP)**: [NLTK](https://www.nltk.org/)
- **Machine Learning**: [Scikit-Learn](https://scikit-learn.org/)
- **Data Visualization**: [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/)
- **Interactive Environment**: [Jupyter Notebook](https://jupyter.org/)

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/vishnukonjeti/ml-practice-portfolio.git
cd ml-practice-portfolio
```

### 2. Set Up a Virtual Environment
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Any Project Notebook
Navigate to the project folder of your choice or open Jupyter Notebook:
```bash
jupyter notebook
```

---

## 📄 License

This repository is licensed under the [MIT License](LICENSE).
