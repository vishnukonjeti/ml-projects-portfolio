# Wine Quality Prediction

This project builds an end-to-end Machine Learning classification system in Python to evaluate and predict whether red wine is of **Good Quality** or **Bad Quality** based on its physicochemical laboratory tests.

---

## 📌 Project Overview

- **Domain**: Food & Beverage / Quality Assurance / Chemical Analytics
- **Problem Type**: Supervised Binary Classification (Binarized Rating)
- **Algorithm**: Random Forest Classifier (`RandomForestClassifier`)
- **Primary Goal**: Classify red wine samples into Good Quality (`1`) or Bad Quality (`0`) to automate sensory score prediction using objective chemical measurements.

---

## 📊 Dataset Details

The dataset used is the **Red Wine Quality Dataset** (Cortéz et al.):

- **File**: `wine_quality_dataset.csv`
- **Total Records**: 1,599 samples
- **Missing Values**: 0 (complete dataset)
- **Features**: 11 physicochemical predictor variables
  1. `fixed acidity`: Concentration of non-volatile organic acids (tartaric acid) in $\text{g/dm}^3$
  2. `volatile acidity`: Concentration of acetic acid in $\text{g/dm}^3$ (high levels cause an unpleasant vinegar taste)
  3. `citric acid`: Citric acid concentration in $\text{g/dm}^3$ (adds freshness and flavor)
  4. `residual sugar`: Amount of natural sugar remaining after fermentation in $\text{g/dm}^3$
  5. `chlorides`: Amount of salt in the wine ($\text{NaCl}$) in $\text{g/dm}^3$
  6. `free sulfur dioxide`: Free $\text{SO}_2$ preventing microbial growth and oxidation in $\text{mg/dm}^3$
  7. `total sulfur dioxide`: Bound and free $\text{SO}_2$ in $\text{mg/dm}^3$
  8. `density`: Density of wine in $\text{g/cm}^3$ (influenced by alcohol and sugar)
  9. `pH`: Acidity/alkalinity scale (ranges between 3.0 and 4.0 for most wines)
  10. `sulphates`: Potassium sulphate additive ($\text{g/dm}^3$) acting as an antimicrobial/antioxidant
  11. `alcohol`: Percent alcohol content by volume (% vol)
- **Target Variable**:
  - Original `quality`: Sensory quality score between 3 and 8
  - **Binarized Classification Label**:
    - `1` (Good Quality): Score $\ge 7$ (ratings 7 and 8)
    - `0` (Bad Quality): Score $< 7$ (ratings 3, 4, 5, and 6)

---

## ⚙️ Machine Learning Pipeline

1. **Data Ingestion & Integrity Inspection**:
   - Loaded into a Pandas DataFrame `(1599, 12)`.
   - Verified zero missing values across all columns.
   - Evaluated summary statistics (`describe()`).

2. **Exploratory Data Analysis (EDA) & Visualizations**:
   - Evaluated quality score distribution using Seaborn count plots (`catplot`).
   - Analyzed chemical correlations with wine quality via bar plots:
     - **Volatile Acidity vs. Quality**: Negative correlation — higher volatile acidity is associated with poorer wine quality.
     - **Citric Acid vs. Quality**: Positive correlation — higher citric acid levels correlate with higher quality ratings.
   - Constructed a full correlation heatmap (`sns.heatmap`) to evaluate multi-variable linear relationships.

3. **Label Binarization**:
   - Binarized the multi-class quality rating into binary classes:
     $$\text{quality} \ge 7 \rightarrow 1 \quad (\text{Good Quality}), \quad \text{quality} < 7 \rightarrow 0 \quad (\text{Bad Quality})$$

4. **Feature & Target Separation**:
   - Features ($X$): 11 physicochemical predictor attributes.
   - Target ($y$): Binary quality label.

5. **Train-Test Split**:
   - 80% training (1,279 samples) and 20% testing (320 samples) (`test_size=0.2`).
   - `random_state = 3` for reproducible splits.

6. **Model Training**:
   - Trained an ensemble Random Forest Classifier: `RandomForestClassifier()`.

7. **Model Evaluation**:
   - **Training Accuracy**: 100.0% (`1.0`)
   - **Testing Accuracy**: ~94.06% (`0.9406`)

8. **Predictive System**:
   - Built an inference routine that accepts an arbitrary 11-feature measurement tuple, reshapes it to `(1, -1)`, performs model prediction, and outputs a clear classification:
     - `1` $\rightarrow$ *"Good Quality Wine"*
     - `0` $\rightarrow$ *"Bad Quality Wine"*

---

## 📁 Project Structure

```text
Project 5 - wine-quality-prediction/
├── wine-quality-prediction.ipynb   # Main Jupyter notebook with EDA, Random Forest & inference
├── wine_quality_dataset.csv        # Red Wine Quality dataset
└── README.md                       # Project documentation
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
jupyter notebook "wine-quality-prediction.ipynb"
```

Run all cells to execute data exploration, EDA visualizations, Random Forest model training, accuracy evaluation, and sample quality predictions.
