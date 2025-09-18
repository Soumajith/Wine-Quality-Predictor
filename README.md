# Wine Quality Classification using Logistic Regression

## Project Overview

This project uses the **Wine Quality Dataset** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality) to build a **Logistic Regression classifier** that predicts whether a Portuguese wine is of **Good Quality (≥7)** or **Not Good Quality (<7)**.

The dataset consists of **6,497 wines (red & white)** with **13 features** describing physicochemical properties such as acidity, residual sugar, chlorides, sulfur dioxide, pH, sulphates, alcohol, and wine type.

---

## Dataset Summary

* **Shape**: 6,497 rows × 13 columns
* **Target Variable**: Wine Quality (0–10, reframed to binary classification)

  * **Good (≥7) → 1**
  * **Not Good (<7) → 0**
* **No missing values**
* **Highly imbalanced distribution** (majority in quality scores 5–6)

### Example Records (first 5 rows)

| fixed acidity | volatile acidity | citric acid | residual sugar | chlorides | free sulfur dioxide | total sulfur dioxide | density | pH   | sulphates | alcohol | quality | wine\_type |
| ------------- | ---------------- | ----------- | -------------- | --------- | ------------------- | -------------------- | ------- | ---- | --------- | ------- | ------- | ---------- |
| 7.4           | 0.70             | 0.00        | 1.9            | 0.076     | 11.0                | 34.0                 | 0.9978  | 3.51 | 0.56      | 9.4     | 5       | red        |
| 7.8           | 0.88             | 0.00        | 2.6            | 0.098     | 25.0                | 67.0                 | 0.9968  | 3.20 | 0.68      | 9.8     | 5       | red        |
| 7.8           | 0.76             | 0.04        | 2.3            | 0.092     | 15.0                | 54.0                 | 0.9970  | 3.26 | 0.65      | 9.8     | 5       | red        |
| 11.2          | 0.28             | 0.56        | 1.9            | 0.075     | 17.0                | 60.0                 | 0.9980  | 3.16 | 0.58      | 9.8     | 6       | red        |
| 7.4           | 0.70             | 0.00        | 1.9            | 0.076     | 11.0                | 34.0                 | 0.9978  | 3.51 | 0.56      | 9.4     | 5       | red        |

---

## Preprocessing & Model

* **Scaling**: Features standardized using `StandardScaler`.
* **Data Split**: 80% training, 20% testing (stratified).
* **Model**: Logistic Regression with `class_weight="balanced"` to handle class imbalance.
* **Solver**: `liblinear` with `max_iter=1000`.

---

## Results

### Classification Metrics

| Class        | Precision | Recall | F1-score |
| ------------ | --------- | ------ | -------- |
| Not Good (0) | 0.92      | 0.71   | 0.80     |
| Good (1)     | 0.39      | 0.76   | 0.51     |

* **Accuracy**: 71.8%
* **ROC-AUC**: \~0.78

### Confusion Matrix

* True Negatives: High → Model is reliable at detecting **Not Good wines**.
* False Positives: Higher → Model sometimes predicts wines as **Good** when they are not.

### Key Observations

* Dataset is dominated by medium-quality wines (5–6).
* Logistic Regression struggles with **precision for Good wines**, though recall is strong (able to catch most good wines).
* Alcohol and sulphates show stronger correlation with quality.

---

## Visualizations

* Wine Quality Distribution (Red vs White)
* Correlation Heatmap of Features
* Confusion Matrix
* ROC Curve

---

## How to Run

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/wine-quality-logistic.git
   cd wine-quality-logistic
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the script:

   ```bash
   python wine_quality_logistic.py
   ```
4. Outputs:

   * Model evaluation metrics (accuracy, precision, recall, F1-score).
   * Plots (heatmap, distribution, confusion matrix, ROC curve).

---

## Requirements

* Python 3.8+
* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn

Install via:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---
