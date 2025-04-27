#################################################################
Sales Prediction Using XGBoost Regressor
#################################################################

This machine learning project predicts retail item sales
using structured data features.

The model is trained using the XGBoost Regressor,
known for its performance and scalability on
tabular datasets.

#################################################################
Project Structure:
├── salesData.csv         # Input dataset
├── sales_prediction.py   # Main Python script
└── README.md             # This README (bash format)

#################################################################
Project Workflow:
1. Load dataset
2. Handle missing values
3. Encode categorical features
4. Visualize important distributions
5. Split dataset into training and test sets
6. Train XGBoost Regressor model
7. Evaluate performance with R² scores
8. Predict item sales for a given input

#################################################################
Dependencies:
Install the required Python libraries using pip:

pip install pandas numpy matplotlib seaborn scikit-learn xgboost

#################################################################
