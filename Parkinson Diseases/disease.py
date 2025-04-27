import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score

dataD = pd.read_csv('parkinson.csv')
dataD.columns = dataD.columns.str.strip()

# print(dataD.head())
# print(dataD.shape)
# print(dataD.isnull().sum())

# 1--> parkinson's Positive 2--> Parkinson's Negatve 
# print(dataD['status'].value_counts())



# Separating features and target 
x = dataD.drop(columns = ['name', 'status'], axis = 1)
y = dataD['status']

# Splitting the data into training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# Data standarization

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
# print(x_train)
# print(x_test)

# SVM
model = svm.SVC(kernel='linear')
model.fit(x_train, y_train)

# Model evaluation (Training Data)
trainPredict = model.predict(x_train)
trainScore = accuracy_score(y_train, trainPredict)
print("Training Accuracy Score", trainScore)

# Model evaluation (Testing Data)
testPredict = model.predict(x_test)
testScore = accuracy_score(y_test, testPredict)
print("Testing Accuracy Score", testScore)


# Predictive Model

# Example input data (replace with actual values)
input_data = (119.99200, 157.30200, 74.99700, 0.00784, 0.00007, 0.00370, 0.00554, 0.01134,
              0.02182, -4.813031, 0.266482, 0.335590, 0.234513, 0.175678, 0.368674,
              22.05500, 0.414783, 0.855723, 0.654300, 0.220480, 0.197144, 0.123306)

# Convert to numpy array and reshape
input_np = np.asarray(input_data).reshape(1, -1)

# Standardize input data using previously fitted scaler
input_std = scaler.transform(input_np)

# Make prediction
prediction = model.predict(input_std)

if prediction[0] == 1:
    print("The person has Parkinson's Disease.")
else:
    print("The person does NOT have Parkinson's Disease.")


# Copy any row of values from csv file to check