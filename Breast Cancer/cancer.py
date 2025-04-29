import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

df = pd.read_csv('Breast Cancer/data.csv')
# Basic information about the dataset

# print(df.head())    
# print(df.isnull().sum())    
# print(df.describe())
# print(df.info())
# print(df.shape)

encoder =  LabelEncoder()
df['diagnosis'] = encoder.fit_transform(df['diagnosis'])
# print(df.head())


# Separating features and target variable
x = df.drop(['Unnamed: 32', 'id', 'diagnosis'], axis=1)
y = df['diagnosis']
# print(x.shape, y.shape)
# print(x.head())
# print(y.head())

# Splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
scalar = StandardScaler()
x_train = scalar.fit_transform(x_train)
x_test = scalar.transform(x_test)

model1 = LogisticRegression()
model1.fit(x_train, y_train)


model1PredictionTraining = model1.predict(x_train)
model1TrainingAccuracy = accuracy_score(y_train, model1PredictionTraining)
print("Training Accuracy of Logistic Regression:", model1TrainingAccuracy)

model1PredictionTesting = model1.predict(x_test)
model1TestingAccuracy = accuracy_score(y_test, model1PredictionTesting)
print("Testing Accuracy of Logistic Regression:", model1TestingAccuracy)

model2 = SVC(kernel='linear')
model2.fit(x_train, y_train)

model2PredictionTraining = model2.predict(x_train)
model2TrainingAccuracy = accuracy_score(y_train, model2PredictionTraining)
print("Training Accuracy of SVC:", model2TrainingAccuracy)

model2PredictionTesting = model2.predict(x_test)    
model2TestingAccuracy = accuracy_score(y_test, model2PredictionTesting)
print("Testing Accuracy of SVC:", model2TestingAccuracy)


# Predicting the diagnosis for a custom input
# ⌨️ Interactive Prediction Model
# input_data = tuple(map(float, input("Enter 30 feature values separated by space: ").split()))
input_data = (17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189)
input_array = np.asarray(input_data).reshape(1, -1)
input_array_scaled = scalar.transform(input_array)

# Predict with both models
prediction1 = model1.predict(input_array_scaled)
prediction2 = model2.predict(input_array_scaled)

label = {0: 'Benign', 1: 'Malignant'}
print(f"🧠 Logistic Regression Prediction: {label[prediction1[0]]}")
print(f"🤖 SVM (Linear Kernel) Prediction: {label[prediction2[0]]}")

# Take any input row from the dataset and predict its diagnosis
