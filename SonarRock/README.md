# Sonar Rock vs Mine Prediction 🎯

This is a simple Machine Learning project where we build a predictive model to classify sonar signals as either **Rocks** or **Mines**, using Logistic Regression.

## 📊 Dataset

- **Name:** Sonar Dataset  
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/connectionist+bench+(sonar,+mines+vs.+rocks))  
- **Samples:** 208  
- **Features:** 60 numeric features  
- **Label:** 'R' = Rock, 'M' = Mine (Encoded to 0 and 1)

## 🧠 Algorithms Used

- Logistic Regression (Binary Classification)

## 🛠️ Libraries Used

- Pandas
- NumPy
- Scikit-learn

## 📌 Project Workflow

1. Load and preprocess data (Label Encoding for categorical labels)
2. Split the dataset into training and test sets
3. Train the Logistic Regression model
4. Evaluate model using accuracy scores
5. Predict manually by giving a row of input

## ✅ Results

- **Training Accuracy:** `83.42%`
- **Testing Accuracy:** `76.19%`

> 🔍 The model performs decently for the given dataset and shows a clear distinction between Rocks and Mines based on sonar frequencies.

## 📂 Project Structure

