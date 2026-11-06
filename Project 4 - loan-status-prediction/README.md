# Loan Status Prediction

This project builds an end-to-end Machine Learning classification system in Python to automate loan eligibility assessment, predicting whether a loan application should be **Approved** or **Rejected** based on applicant demographic, financial, and credit history details.

---

## 📌 Project Overview

- **Domain**: Banking / Financial Services / Credit Risk Analytics
- **Problem Type**: Supervised Binary Classification
- **Algorithm**: Support Vector Machine (Linear SVM)
- **Primary Goal**: Classify loan applications into Approved (`1` / `Y`) or Rejected (`0` / `N`) to assist financial institutions in making faster, data-driven credit decisions.

---

## 📊 Dataset Details

The dataset used is the **Loan Status Dataset** containing historical loan applicant records:

- **File**: `loan_status_dataset.csv`
- **Total Records**: 614 initial applicants (480 after handling missing values)
- **Features**: 11 diagnostic and socio-economic predictor variables
  1. `Gender`: Gender of the applicant (`Male`: 1, `Female`: 0)
  2. `Married`: Marital status (`Yes`: 1, `No`: 0)
  3. `Dependents`: Number of dependent family members (`0`, `1`, `2`, `3+` mapped to `4`)
  4. `Education`: Applicant education level (`Graduate`: 1, `Not Graduate`: 0)
  5. `Self_Employed`: Employment status (`Yes`: 1, `No`: 0)
  6. `ApplicantIncome`: Applicant monthly income
  7. `CoapplicantIncome`: Co-applicant monthly income
  8. `LoanAmount`: Loan amount in thousands
  9. `Loan_Amount_Term`: Term of the loan in months (e.g., 360, 180)
  10. `Credit_History`: Credit history guideline compliance (`1.0`: Good / Meets guidelines, `0.0`: Bad / Does not meet)
  11. `Property_Area`: Property location type (`Rural`: 0, `Semiurban`: 1, `Urban`: 2)
- **Target Variable**:
  - `Loan_Status`:
    - `1` (`Y`): Loan Approved / Eligible
    - `0` (`N`): Loan Rejected / Not Eligible

---

## ⚙️ Machine Learning Pipeline

1. **Data Ingestion & Missing Value Treatment**:
   - Loaded into a Pandas DataFrame.
   - Inspected initial dimensions `(614, 13)` and identified missing records across `Gender`, `Married`, `Dependents`, `Self_Employed`, `LoanAmount`, `Loan_Amount_Term`, and `Credit_History`.
   - Handled missing values via row removal (`dropna()`), producing 480 complete applicant profiles.

2. **Categorical Encoding & Feature Preprocessing**:
   - Target variable label encoding: `{'N': 0, 'Y': 1}`.
   - Standardized `Dependents` by replacing `'3+'` with `4`.
   - Converted categorical attributes to numerical representations:
     - `Gender`: `{'Male': 1, 'Female': 0}`
     - `Married`: `{'No': 0, 'Yes': 1}`
     - `Education`: `{'Not Graduate': 0, 'Graduate': 1}`
     - `Self_Employed`: `{'No': 0, 'Yes': 1}`
     - `Property_Area`: `{'Rural': 0, 'Semiurban': 1, 'Urban': 2}`

3. **Exploratory Data Analysis (EDA)**:
   - Visualized loan approval trends using Seaborn count plots:
     - **Education vs. Loan Status**: Higher approval proportion observed for graduates.
     - **Marital Status vs. Loan Status**: Married applicants exhibit higher likelihood of approval.
     - **Gender vs. Loan Status**: Evaluated distribution across gender groups.

4. **Feature & Target Separation**:
   - Features ($X$): 11 predictor columns (excluding non-predictive `Loan_ID` and target `Loan_Status`).
   - Target ($y$): `Loan_Status` cast as integer labels.

5. **Stratified Train-Test Split**:
   - 90% training (432 samples) and 10% testing (48 samples) (`test_size=0.1`).
   - Stratified sampling (`stratify=y`) to maintain approval/rejection proportions in both splits.
   - `random_state = 2` for deterministic reproducibility.

6. **Model Training**:
   - Trained a Support Vector Classifier with a linear kernel: `svm.SVC(kernel='linear')`.

7. **Model Evaluation**:
   - **Training Accuracy**: ~79.86%
   - **Testing Accuracy**: ~83.33%

8. **Predictive System**:
   - Built an end-to-end inference routine that takes an applicant's profile tuple, reshapes it to `(1, -1)`, runs `classifier.predict()`, and outputs a clear recommendation:
     - `1` $\rightarrow$ *"The person is eligible for loan"*
     - `0` $\rightarrow$ *"The person is not eligible for loan"*

---

## 📁 Project Structure

```text
Project 4 - loan-status-prediction/
├── loan-status-prediction.ipynb   # Main Jupyter notebook with EDA, preprocessing & SVM model
├── loan_status_dataset.csv        # Loan applicant dataset
└── README.md                      # Project documentation
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
jupyter notebook "loan-status-prediction.ipynb"
```

Run all cells to execute data exploration, categorical encoding, SVM classifier training, accuracy evaluation, and sample applicant predictions.
