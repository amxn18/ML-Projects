# Diabetic-Prediction
# 🩺 Diabetes Prediction using SVM

This is a simple Machine Learning project that predicts whether a person is diabetic or not based on their health parameters. The model is trained using the Support Vector Machine (SVM) algorithm on the popular **PIMA Indian Diabetes Dataset**.

---

## 📊 Dataset

- **Name:** PIMA Indian Diabetes Dataset  
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes)  
- **Samples:** 768  
- **Features:** 8 numeric medical attributes (e.g. glucose, BMI, age, etc.)  
- **Target:** `0` (Non-diabetic), `1` (Diabetic)

---

## 🔧 Tech Stack

- Python 🐍
- Pandas & NumPy
- Scikit-learn (SVM, train_test_split, StandardScaler, accuracy_score)

---

## 📌 Workflow

1. **Data Loading**  
   Load the dataset and explore the structure.

2. **Preprocessing**  
   - Separate features and target.
   - Standardize the feature values using `StandardScaler`.

3. **Train-Test Split**  
   Split the dataset using an 80-20 ratio with stratified sampling.

4. **Model Training**  
   Train a **Support Vector Classifier (SVC)** with a **linear kernel**.

5. **Model Evaluation**  
   Evaluate model performance on both training and testing sets using **accuracy score**.

6. **Prediction System**  
   Make predictions on new input data.

---

## ✅ Results

| Kernel     | Training Accuracy | Testing Accuracy |
|------------|-------------------|------------------|
| Linear     | ~78.6%            | ~77.2%           |
| RBF        | ~84%              | ~82%             |

👉 The **RBF kernel** performs better for this non-linearly separable data.

---

## ⚠️ Note on Warnings

You may encounter:
