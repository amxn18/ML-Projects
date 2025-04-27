import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

heartData = pd.read_csv('heartData.csv')

# Data Analysis

# print(heartData.isnull().sum())
# print(heartData['target'].value_counts()) 
# 1-->Defective Heart 0-->Healthy Heart

# Seprating target variable and features
x = heartData.drop(columns='target', axis=1)
y = heartData['target']

# Splitting dataset into training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# Training Model
model = LogisticRegression()
model.fit(x_train, y_train)

# Accuracy Score(Training Score)
trainPrediction = model.predict(x_train)
trainScore = accuracy_score(trainPrediction, y_train)
print("Training Score:", trainScore)

# Accuracy Score(Testing Score)
testPrediction = model.predict(x_test)
testScore = accuracy_score(testPrediction, y_test)
print("Testing Score:", testScore)

# Predictive model
inputData = ()
inputDataArray = np.asarray(inputData)
reshapedData = inputDataArray.reshape(1,-1)

prediction = model.predict(reshapedData)
# print(prediction)

if(prediction[0] == 1): 
    print("Person has a Defective Heart")
else:
    print("Person has a Healthy Heart")

# Copy and paste row of dataset in inputData to check 
