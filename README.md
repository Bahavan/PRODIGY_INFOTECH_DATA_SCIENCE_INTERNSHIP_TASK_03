# PRODIGY_INFOTECH_DATA_SCIENCE_INTERNSHIP_TASK_03

# Task-03: Decision Tree Classifier for Customer Purchase Prediction

## Task Description
Build a decision tree classifier to predict whether a customer will purchase a product or service based on their demographic and behavioral data. Use a dataset such as the **Bank Marketing dataset** from the **UCI Machine Learning Repository**.

## Dataset
You can access the Bank Marketing dataset from UCI Machine Learning Repository here:
[Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)

## Instructions
1. Download the dataset from the provided link.
2. Load the dataset using `pandas`.
3. Perform **Data Preprocessing**:
   - Handle missing values.
   - Encode categorical variables.
   - Scale numerical features if necessary.
4. **Train a Decision Tree Classifier**:
   - Split the data into training and testing sets.
   - Train a decision tree model.
   - Evaluate the model performance using accuracy, precision, recall, and F1-score.
5. **Visualize the Decision Tree**:
   - Plot the tree structure using `graphviz` or `matplotlib`.

## Requirements
- Python 3.x
- Libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `graphviz` (optional for tree visualization)

## Example Code
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("bank.csv", sep=";")  # Use appropriate delimiter

# Encode categorical variables
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Split dataset
X = df.drop(columns=["y"])  # Assuming 'y' is the target variable
y = df["y"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Plot decision tree
plt.figure(figsize=(12, 8))
plot_tree(model, feature_names=X.columns, class_names=["No", "Yes"], filled=True)
plt.show()
