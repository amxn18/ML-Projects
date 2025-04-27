# 🔥 Calories Burnt Prediction using Machine Learning

This project uses Machine Learning to predict the number of calories burnt during exercise based on physiological parameters. It uses the **XGBoost Regressor** model, trained on combined data from two datasets: `exercise.csv` and `calories.csv`.

---

## 📌 Project Overview

In this project, we:

- Explored and merged two datasets (`exercise.csv` + `calories.csv`)
- Performed data visualization and basic analysis
- Encoded categorical data (Gender)
- Built and trained an ML regression model using **XGBoost**
- Evaluated the model with R² Score on both training and testing datasets
- Created an input-based prediction system for real-time calorie estimation

---

## 📊 Dataset Description

Two datasets were used:

- `exercise.csv`: Contains features like `User_ID`, `Gender`, `Age`, `Height`, `Weight`, `Duration`, `Heart_Rate`, `Body_Temp`
- `calories.csv`: Contains `User_ID` and `Calories` burnt

They are merged on the basis of matching `User_ID` values.

---

## 🛠️ Tech Stack

- **Python**
- **Pandas** – Data Handling
- **Seaborn / Matplotlib** – Visualization
- **scikit-learn** – Model evaluation
- **XGBoost** – Machine Learning model

---

## 📌 Features Used

- `Gender` (encoded as 0 or 1)
- `Age`
- `Height` (in cm)
- `Weight` (in kg)
- `Duration` (in minutes)
- `Heart Rate`
- `Body Temperature` (in °C)

---

## 📈 Model Training & Evaluation

- Model Used: `XGBRegressor`
- Data split: 80% training, 20% testing
- Evaluation Metric: **R² Score**

###