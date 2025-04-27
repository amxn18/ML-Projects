import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

ccData = pd.read_csv('creditcard.csv')
# All features are already converted to numerical value using principal component analysis as cc details are sensative

# Data analysis

# print(ccData.head())
# print(ccData.isnull().sum())

# 0--> Legit transactions 1--> Fraudlent Transactions 
# print(ccData['Class'].value_counts())  
# 0    284315
# 1       492

# The Data in unbalanced 
legit = ccData[ccData.Class == 0]
fraud = ccData[ccData.Class == 1]

# Under Sampling
legitSample = ccData.sample(n=492)
# print(fraud.shape)
# print(legitSample.shape)

newCCData = pd.concat([legitSample,fraud], axis =0)
# print(newCCData['Class'].value_counts())

# Seprating into features and labels
x = newCCData.drop(columns='Class', axis = 1)
y = newCCData['Class']


# Splitting the data into training and test
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.1, random_state=2)

# Logistic Regression Model
model = LogisticRegression()
model.fit(x_train, y_train)

# Accuracy Score(Training)
trainPredict = model.predict(x_train)
trainScore = accuracy_score(trainPredict, y_train)
print("Accuracy Score(Training):", trainScore)

# Accuracy Score(Testing)
testPredict = model.predict(x_test)
testScore = accuracy_score(testPredict, y_test)
print("Accuracy Score(Test Data):", testScore)

# Predictive Model
# Predictive System
inputData = (0.0, 0.0, 2.53634673796914, 1.37815522497476, 0.0, 0.0, 0.0, 0.0, 
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # Replace with any row values

inputDataArray = np.asarray(inputData).reshape(1, -1)
prediction = model.predict(inputDataArray)

if prediction[0] == 1:
    print(" Transaction is Fraudulent")
else:
    print(" Transaction is Legitimate")


# ou can copy a row from creditcard.csv (excluding the Class label) to test a real value.
# The dataset has 30 features (V1 to V28 + Time, Amount) → total 30 values in each input row.