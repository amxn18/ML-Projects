# Credit Card Fraud Detection 🏦🔍
DataSet: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
## 📌 Objective
To detect fraudulent credit card transactions using machine learning (Logistic Regression). The dataset is highly imbalanced, so under-sampling was used to handle the class distribution.

## 📁 Dataset
- File: `creditcard.csv`
- Features already transformed using PCA for confidentiality.
- Class Distribution:
  - Legit: 284,315 transactions
  - Fraudulent: 492 transactions

## ⚙️ Workflow

1. **Data Loading & Inspection**
   - Loaded the CSV using `pandas`
   - Checked for null values (none found)

2. **Handling Imbalanced Dataset**
   - Original dataset was highly imbalanced
   - Performed **under-sampling** to balance the classes:
     - Took a random sample of 492 legit transactions
     - Combined with all 492 fraud transactions

3. **Feature & Label Separation**
   - `x` ➝ features (all columns except `Class`)
   - `y` ➝ target label (`Class`: 0 → Legit, 1 → Fraud)

4. **Train-Test Split**
   - Split data using 90% for training, 10% for testing

5. **Modeling**
   - Trained a **Logistic Regression** model

6. **Evaluation**
   - Accuracy (Training): `0.938`
   - Accuracy (Testing): `0.934` *(values may vary run to run)*

## 🧠 Libraries Used
- `pandas`
- `numpy`
- `sklearn.model_selection` ➝ `train_test_split`
- `sklearn.linear_model` ➝ `LogisticRegression`
- `sklearn.metrics` ➝ `accuracy_score`

## 📝 Notes
- Since this is an imbalanced dataset, other metrics like **precision, recall, F1-score** and **confusion matrix** are also useful (but not covered here).
- You can also experiment with:
  - **SMOTE** (over-sampling minority class)
  - **RandomForestClassifier**, **XGBoost**, etc., for better performance

