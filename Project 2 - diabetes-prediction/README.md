# Diabetes Prediction

This project builds an end-to-end Machine Learning classification system in Python to predict whether a patient is **Diabetic** or **Non-Diabetic** based on clinical diagnostic and physiological measurements.

---

## 📌 Project Overview

- **Domain**: Healthcare / Medical Diagnostics / Predictive Analytics
- **Problem Type**: Supervised Binary Classification
- **Algorithm**: Support Vector Machine (Linear SVM)
- **Primary Goal**: Predict the presence of diabetes (`Outcome`: `0` or `1`) using diagnostic predictor variables.

---

## 📊 Dataset Details

The dataset used is the **Pima Indians Diabetes Database** (originally from the National Institute of Diabetes and Digestive and Kidney Diseases):

- **File**: `diabetes.csv`
- **Total Records**: 768 patient samples
- **Features**: 8 diagnostic predictor variables
  1. `Pregnancies`: Number of times pregnant
  2. `Glucose`: Plasma glucose concentration after 2 hours in an oral glucose tolerance test (mg/dL)
  3. `BloodPressure`: Diastolic blood pressure (mm Hg)
  4. `SkinThickness`: Triceps skin fold thickness (mm)
  5. `Insulin`: 2-Hour serum insulin (mu U/ml)
  6. `BMI`: Body Mass Index (weight in kg / (height in m)²)
  7. `DiabetesPedigreeFunction`: Diabetes pedigree function (genetic risk score)
  8. `Age`: Patient age in years
- **Target Variable**:
  - `0` : Non-Diabetic (500 instances, ~65.1%)
  - `1` : Diabetic (268 instances, ~34.9%)

---

## ⚙️ Machine Learning Pipeline

1. **Data Ingestion & Statistical Exploration**:
   - Loaded the dataset into a Pandas DataFrame.
   - Evaluated data shape `(768, 9)` and summary statistics (`mean`, `std`, percentiles).
   - Inspected class balance and analyzed feature means grouped by class (`groupby('Outcome').mean()`), observing significantly higher average glucose, BMI, age, and pregnancy counts among diabetic patients.

2. **Feature & Target Separation**:
   - Features ($X$): 8 clinical predictor columns.
   - Target ($y$): `Outcome` column.

3. **Data Standardization**:
   - Standardized the predictor features using `StandardScaler` from `sklearn.preprocessing`.
   - Scaled features to zero mean and unit variance ($z = \frac{x - \mu}{\sigma}$), which is essential for distance-based margin classifiers such as Support Vector Machines.

4. **Stratified Train-Test Split**:
   - `test_size = 0.2` (80% training: 614 samples, 20% testing: 154 samples).
   - `stratify = y` to maintain the identical outcome class ratio across both splits.
   - `random_state = 2` for deterministic and reproducible splits.

5. **Model Training**:
   - Trained a Support Vector Classifier with a linear kernel: `svm.SVC(kernel='linear')`.

6. **Model Evaluation**:
   - **Training Accuracy**: ~78.66%
   - **Testing Accuracy**: ~77.27%

7. **Predictive System**:
   - Implemented an end-to-end inference routine that accepts arbitrary patient diagnostic values as an input tuple, converts and reshapes it to `(1, -1)`, scales it using the fitted `StandardScaler`, and outputs whether the individual is **Diabetic** or **Not Diabetic**.

---

## 📁 Project Structure

```text
Project 2 - diabetes-prediction/
├── diabetes-prediction.ipynb   # Main Jupyter notebook with EDA, preprocessing & SVM model
├── diabetes.csv                # Pima Indians Diabetes dataset
└── README.md                   # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.9+ installed and dependencies from the repository's root installed:

```bash
pip install -r ../requirements.txt
```

### 2. Launch the Notebook
Open the project directory in VS Code or run Jupyter directly:

```bash
jupyter notebook "diabetes-prediction.ipynb"
```

Run all cells to replicate data exploration, feature scaling, SVM training, accuracy evaluation, and the predictive system.

