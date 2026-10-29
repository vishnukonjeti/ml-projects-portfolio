# SONAR Rock vs Mine Prediction

This project builds a Python-based machine learning system that predicts whether a sonar signal corresponds to a rock or a mine. The model uses Logistic Regression, a simple and effective binary classification algorithm for this type of dataset.

## Problem Statement
The dataset contains sonar signal readings collected from different angles. Each row represents a sonar sample, and the target variable indicates whether the object is:

- R = Rock
- M = Mine

The goal is to train a model that can generalize from these sonar features and classify new observations accurately.

## Project Workflow
1. Load the SONAR dataset
2. Separate features and target labels
3. Split the data into training and testing sets
4. Standardize the feature values
5. Train a Logistic Regression model
6. Evaluate the model using accuracy, precision, recall, and confusion matrix

## Model Used
- Logistic Regression
- Binary classification
- Suitable for learning a decision boundary between rock and mine sonar patterns

## Requirements
Install the project dependencies from the repository-level requirements file:

```bash
pip install -r requirements.txt
```

## Run the Notebook
```bash
jupyter notebook
```

Then open the `notebook.ipynb` file inside this project folder and run all cells.

## Expected Outcome
The system will produce a trained classifier that can estimate whether a sonar observation is more likely to correspond to a rock or a mine based on the learned patterns from the dataset.
