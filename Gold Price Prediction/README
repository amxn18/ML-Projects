# Gold Price Prediction Using Random Forest Regressor

This project predicts the price of gold (GLD) based on financial indicators like SPX, USO, SLV, and EUR/USD using a machine learning model — Random Forest Regressor.

## 📁 Dataset

- File: `goldPrice.csv`
- Features:
  - `Date`
  - `SPX`: S&P 500 Index
  - `GLD`: Gold ETF price (target)
  - `USO`: Crude oil price ETF
  - `SLV`: Silver ETF price
  - `EUR/USD`: Euro to USD exchange rate

## 📊 Correlation Heatmap

A heatmap is generated to visualize correlations between features. Strong correlation helps identify useful predictors for gold price.

## 🧠 ML Model: Random Forest Regressor

Steps:
1. Data cleaning & preprocessing (removed `Date` column)
2. Feature-target split (`x`: other features, `y`: `GLD`)
3. Train-test split (80-20)
4. Model training using `RandomForestRegressor`
5. Performance evaluation using R² Score

## 🧪 Model Performance

R² Score is used to evaluate:
- Training data performance
- Testing data performance

## 📈 Predict Custom Input

To predict the gold price using your custom input:

```python
input_data = [SPX, USO, SLV, EUR/USD]  # example: [1450, 55.0, 16.0, 1.12]
input_array = np.array(input_data).reshape(1, -1)
predicted_price = model.predict(input_array)
