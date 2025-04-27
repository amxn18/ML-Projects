import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

goldData = pd.read_csv('goldPrice.csv')
# print(goldData.head())
# print(goldData.isnull().sum())

# Correlation 1) Positive Correlation(~1) 2) Negative Coorelation(~-1)
correlation = goldData.drop(columns=['Date']).corr()
plt.figure(figsize=(6,6))
sns.heatmap(correlation, cbar=True, square=True, fmt = '.1f', annot=True, annot_kws={'size': 8}, cmap = 'Blues')
# plt.show()
print(correlation['GLD'])


# Splitting the data
x = goldData.drop(columns = ['Date', 'GLD'], axis= 1)
y = goldData['GLD']
# print(x)
# print(y)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=2)

# Random Forest Regressor 
model = RandomForestRegressor()
model.fit(x_train, y_train)

# Model Evaluation(Training Data)
traingPrediction = model.predict(x_train)
trainingScore = metrics.r2_score(traingPrediction, y_train)
print("Training Score:", trainingScore)

# Model Evaluation(Testing Data)
testingPrediction = model.predict(x_test)
testScore = metrics.r2_score(testingPrediction, y_test)
print("Test Score:", testScore)

# Predicting the Gold price for a custom input
print(x.columns) 
input_data = [1450, 55.0, 16.0, 1.12]  # example values for [SPX, USO, SLV, EUR/USD]
input_array = np.array(input_data).reshape(1, -1)
predicted_price = model.predict(input_array)
print("Predicted Gold Price (GLD):", predicted_price[0])








