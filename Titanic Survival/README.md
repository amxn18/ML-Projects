# Titanic Survival Prediction 🚢

This is a Machine Learning project to predict whether a passenger survived the Titanic shipwreck based on features like age, class, sex, fare, and more. Two models are used:

- Logistic Regression
- XGBoost Classifier

---

## 📁 Dataset Used

We use the classic Titanic dataset from Kaggle: `train.csv`.

---

## 🛠️ Libraries Required

Run the following to install dependencies:

pip install pandas numpy matplotlib seaborn scikit-learn xgboost

---

## 📌 Steps Performed

1. Data Loading
   - Load train.csv using pandas

2. Data Cleaning
   - Dropped 'Cabin' column due to too many missing values
   - Replaced missing 'Age' values with the mean
   - Filled missing 'Embarked' values with the mode

3. Visualization
   - Used seaborn to plot 'Survived' count and 'Sex' vs 'Survived'

4. Label Encoding
   - Converted 'Sex': male → 0, female → 1
   - Converted 'Embarked': S → 0, C → 1, Q → 2

5. Feature & Target Separation
   - Features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
   - Target: Survived

6. Model Training
   - Logistic Regression
   - XGBoost Classifier

7. Evaluation
   - Measured accuracy on training and testing datasets

---

## 🎯 Accuracy Results

Logistic Regression:
- Training Accuracy: ~0.8075
- Testing Accuracy:  ~0.7821

XGBoost Classifier:
- Training Accuracy: ~0.9747
- Testing Accuracy:  ~0.7821

Note: XGBoost may be overfitting slightly.

---

## 🧪 Predictive Input (using XGBoost model)

Sample input (converted):

(3, 1, 22.0, 1, 0, 7.25, 0)

Where:
- Pclass = 3
- Sex = 1 (female)
- Age = 22.0
- SibSp = 1
- Parch = 0
- Fare = 7.25
- Embarked = 0 (Southampton)

Run:

prediction = model2.predict([input_data])

If output is 1 → ✅ Survived  
If output is 0 → ❌ Did not survive

---

## 📂 Project Structure

Titanic Survival/
├── train.csv  
├── titanic_model.py       # Main code  
├── README.md              # This file  
└── __pycache__/           # Ignore  

---

## 👨‍💻 Author

- Name: Aman  
- Built with: Python, Scikit-learn, XGBoost, Seaborn, Matplotlib

---

## 🚀 Future Ideas

- Use titles (Mr., Mrs., etc.) from names
- Hyperparameter tuning for XGBoost
- Build web interface with Streamlit or Flask

---

## ✅ Done!

Thanks for exploring this project! 🚀 Feel free to fork, clone, improve, or ask questions.
