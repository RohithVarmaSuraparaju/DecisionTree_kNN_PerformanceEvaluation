Name: Rohith Varma Suraparaju
ID: 700771851

# DecisionTree_kNN_PerformanceEvaluation
# Machine Learning with scikit-learn — Decision Trees, kNN, and Performance Evaluation

This repository contains three Jupyter notebooks demonstrating fundamental supervised learning techniques using scikit-learn. Each notebook is well-commented with human-friendly explanations to help you learn how to build, evaluate, and interpret machine learning models.

---

## 📘 Notebooks Overview

### 1. `DecisionTree_sklearn.ipynb`
Goal: Explore Decision Trees on the Iris dataset.  
Steps:
- Load the Iris dataset and split into training/testing sets.
- Train `DecisionTreeClassifier` models with `max_depth = 1, 2, 3`.
- Report training and test accuracy for each depth.
- Discuss underfitting vs. overfitting using accuracy gaps and tree complexity.
- Visualize the trained decision trees and their decision boundaries.

Key insights:
- Shallow trees (depth = 1) underfit: both train & test accuracy are low.
- Moderately deep trees can provide good generalization.
- Very deep trees may overfit if train accuracy ≫ test accuracy.

---

### 2. `kNN_Classification_sklearn.ipynb`
Goal: Demonstrate k-Nearest Neighbors (kNN) with different values of `k`.  
Steps:
- Use only two features from Iris (sepal length & sepal width) for visualization.
- Train kNN models with `k = 1, 3, 5, 10`.
- Plot decision boundaries for each `k` to visualize how neighborhood size affects classification.
- Comment on how boundaries change as `k` increases.

Key insights:
- `k=1`: highly flexible, jagged boundaries → low bias, high variance (overfitting risk).
- `k=3,5`: smoother boundaries, better generalization.
- `k=10`: very smooth, may underfit if `k` is too large.

---

### 3. `Performance_Evaluation.ipynb`
Goal: Evaluate classifiers with multiple performance metrics.  
Steps:
- Train a kNN classifier (`k=5`) on the Iris dataset.
- Compute and plot the **confusion matrix**.
- Produce a **classification report** (precision, recall, F1, accuracy).
- Plot **ROC curves** (one-vs-rest for multiclass) and compute **AUC** values.

Key insights:
- Confusion matrices reveal *which classes are confused*.
- Precision/Recall/F1 give more nuanced views than accuracy (especially for imbalanced data).
- ROC/AUC visualize trade-offs between sensitivity and specificity; for multiclass, use one-vs-rest.

---

## ⚙️ Setup Instructions

1. Clone or download this repository.
2. Install dependencies (Python 3.8+ recommended):
pip install -r requirements.txt
Typical requirements:
numpy
pandas
matplotlib
scikit-learn
Open a notebook and run cells step-by-step.
