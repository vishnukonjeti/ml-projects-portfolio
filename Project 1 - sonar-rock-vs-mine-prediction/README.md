# SONAR Rock vs. Mine Prediction

This project builds an end-to-end Machine Learning classification system in Python to predict whether a submerged object detected by sonar signals is a **Rock** or a **Mine (Metal Cylinder)**.

---

## 📌 Project Overview

- **Domain**: Oceanography / Defense / Signal Processing
- **Problem Type**: Supervised Binary Classification
- **Algorithm**: Logistic Regression
- **Primary Goal**: Classify sonar return signals into Rock (`R`) or Mine (`M`) based on frequency energy patterns.

---

## 📊 Dataset Details

The dataset used is the standard **Sonar, Mines vs. Rocks** dataset (collected by Gorman and Sejnowski):

- **File**: `sonar_data.csv`
- **Total Records**: 208 samples
- **Features**: 60 continuous numerical attributes (each representing energy in a particular frequency band integrated over time)
- **Target Variable**:
  - `R` : Rock (97 instances)
  - `M` : Mine / Metal cylinder (111 instances)

---

## ⚙️ Machine Learning Pipeline

1. **Data Ingestion & Inspection**:
   - Loaded into a Pandas DataFrame (`header=None`).
   - Analyzed statistical properties (`mean`, `std`, percentiles) and confirmed zero missing values.
   - Evaluated class distribution and feature means grouped by class.

2. **Feature & Target Separation**:
   - Features ($X$): First 60 columns.
   - Target ($Y$): 60th index column containing labels (`R` and `M`).

3. **Stratified Train-Test Split**:
   - `test_size = 0.1` (90% training, 10% test).
   - `stratify = Y` to preserve the proportion of rocks and mines across both sets.
   - `random_state = 1` for reproducible results.

4. **Model Training**:
   - Trained a `LogisticRegression` classifier on the training set.

5. **Model Evaluation**:
   - **Training Accuracy**: ~83.42%
   - **Testing Accuracy**: ~76.19%

6. **Predictive System**:
   - Built a custom inference pipeline that takes an arbitrary tuple of 60 sonar frequency readings, reshapes it to `(1, -1)`, and outputs whether the object is a **Rock** or a **Mine**.

---

## 📁 Project Structure

```text
sonar-rock-vs-mine-prediction/
├── Rock vs Mine Prediction.ipynb   # Main Jupyter notebook with data analysis & model
├── sonar_data.csv                  # Sonar dataset
└── README.md                       # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.9+ installed and a virtual environment set up with the repository's dependencies:

```bash
pip install -r ../requirements.txt
```

### 2. Launch the Notebook
Open the project directory in VS Code or run Jupyter directly:

```bash
jupyter notebook "Rock vs Mine Prediction.ipynb"
```

Run all cells to replicate the data exploration, model training, accuracy evaluation, and test predictions.
