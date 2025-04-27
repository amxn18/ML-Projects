# 🍷 Wine Quality Prediction using Machine Learning

This project aims to predict the quality of red wine based on various physicochemical properties using a Machine Learning model. We used the **Random Forest Classifier** to classify wine as **Good Quality** (quality >= 7) or **Bad Quality** (quality < 7).

---

## 📁 Dataset

- Source: UCI Machine Learning Repository  
- File: `winequality-red.csv`  
- Delimiter: `;` (semicolon)

---

## 🔍 Features Used

- Fixed acidity
- Volatile acidity
- Citric acid
- Residual sugar
- Chlorides
- Free sulfur dioxide
- Total sulfur dioxide
- Density
- pH
- Sulphates
- Alcohol  
- **Target**: `quality` (converted into binary: Good or Bad)

---

## 📊 Exploratory Data Analysis (EDA)

- Countplot of wine quality
- Barplots:
  - Volatile acidity vs. Quality
  - Citric acid vs. Quality
  - Residual sugar vs. Quality
- Correlation Heatmap

---

## 🧠 Model Used

### ✅ Random Forest Classifier

We chose Random Forest over other classifiers like Logistic Regression or SVM because:

- It's **robust to outliers and noisy data**.
- Can **automatically handle feature importance**.
- Works great **without feature scaling**.
- Performs well on **non-linear** and **imbalanced datasets**.
- Gives **high accuracy** with minimal tuning.

Other classifiers like:
- **Logistic Regression** assumes a linear relationship (not ideal here).
- **SVM** needs more parameter tuning and scaling and may not scale well with bigger datasets.

---

## 🧪 Evaluation

- Accuracy (Training): ~93%
- Accuracy (Testing): ~91%
- Model performs well on both training and unseen data.

---

## 🔮 Final Prediction

We created an input interface to test any custom wine properties and get a quality prediction:
```python
inputData = (7.4, 0.7, 0, 1.9, 0.076, 11, 34, 0.9978, 3.51, 0.56, 9.4)

---

### 📌 Summary: Why Random Forest?

| Model                 | Pros                                                    | Why Not Used |
|----------------------|---------------------------------------------------------|--------------|
| **Random Forest**     | Handles non-linearity, feature importance, no scaling needed, robust | ✅ Best Fit |
| Logistic Regression  | Fast, interpretable, but assumes linearity              | ❌ Wine data is nonlinear |
| SVM                  | Powerful for small clean datasets                       | ❌ Needs tuning & scaling |
| KNN                  | Easy to understand                                      | ❌ Slow with large datasets |
| Decision Tree        | Good but prone to overfitting                           | ❌ RF solves this with multiple trees |

---

Want me to customize the title, or generate a Streamlit app layout too?