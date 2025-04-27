import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso
from sklearn import metrics

# Load the dataset
carData = pd.read_csv('carData.csv')

# Encoding categorical variables
carData['Transmission'] = carData['Transmission'].replace({'Manual': 1, 'Automatic': 0})
carData['Seller_Type'] = carData['Seller_Type'].replace({'Dealer': 0, 'Individual': 1})
carData['Fuel_Type'] = carData['Fuel_Type'].replace({'Petrol': 0, 'Diesel': 1, 'CNG': 2})

# Split data into features and target
x = carData.drop(columns=['Selling_Price', 'Car_Name'], axis=1)
y = carData['Selling_Price']

# Split into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=2)

# Model training
model = XGBRegressor()
model.fit(x_train, y_train)

# Model evaluation
trainPredict = model.predict(x_train)
trainScore = metrics.r2_score(y_train, trainPredict)
print("Training R² Score:", trainScore)

# Plot actual vs predicted for training data
plt.scatter(y_train, trainPredict)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title("Training Data: Actual vs Predicted Prices")
plt.show()

# Test set prediction
testPredict = model.predict(x_test)
testScore = metrics.r2_score(y_test, testPredict)
print("Test R² Score:", testScore)

# Plot actual vs predicted for test data
plt.scatter(y_test, testPredict)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title("Test Data: Actual vs Predicted Prices")
plt.show()


print("Let's predict your car price!")

year = int(input("Enter Year of Purchase: "))
present_price = float(input("Enter Present Price (in Lakhs): "))
kms_driven = int(input("Enter Kilometers Driven: "))
fuel_type = input("Enter Fuel Type (Petrol/Diesel/CNG): ").strip().lower()
seller_type = input("Enter Seller Type (Dealer/Individual): ").strip().lower()
transmission = input("Enter Transmission Type (Manual/Automatic): ").strip().lower()
owner = int(input("Enter Number of Previous Owners (0/1/3): "))


if fuel_type == 'petrol':
    fuel_type_encoded = 0
elif fuel_type == 'diesel':
    fuel_type_encoded = 1
else:
    fuel_type_encoded = 2  

seller_type_encoded = 0 if seller_type == 'dealer' else 1
transmission_encoded = 1 if transmission == 'manual' else 0


final_input = np.array([[year, present_price, kms_driven,
                         fuel_type_encoded, seller_type_encoded,
                         transmission_encoded, owner]])


columns = x.columns.tolist()
input_df = pd.DataFrame(final_input, columns=columns)
print("Data received for prediction:")
print(input_df)

predicted_price = model.predict(final_input)
print("Predicted Car Price: ₹ {predicted_price[0]:.2f} Lakhs")
