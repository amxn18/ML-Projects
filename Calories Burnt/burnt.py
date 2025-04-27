import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

data1 = pd.read_csv('exercise.csv')
data2 = pd.read_csv('calories.csv')

# print(data1.head())
# print(data2.head())

# Combining Two different Dataframes
dataC = pd.concat([data1, data2['Calories']],axis=1)

# Data Analysis

# print(dataC.head())
# print(dataC.shape)
# print(dataC.info())
# print(dataC.describe())
# print(dataC.isnull().sum())

# Data Visualisation

# sns.set_theme()
sns.countplot(x=dataC['Gender'])  # Well Distributed
sns.displot(dataC['Age'])
sns.displot(dataC['Height'])
# plt.show()

# Encoding Text Data to Numerical Values
dataC.replace({'Gender': {'male':0, 'female':1}}, inplace=True)
# print(dataC.head())

# Separating features and targets
x = dataC.drop(columns=['User_ID','Calories'],axis=1)
y = dataC['Calories']

# Splitting dataset into training and testing dataset
x_train, x_test, y_train, y_test = train_test_split(x , y, test_size=0.2, random_state=2)


# XGBREgressor Model
model = XGBRegressor()
model.fit(x_train, y_train)

# Model Evaluation (Training Data)
trainPredict = model.predict(x_train)
r2TrainingScore = metrics.r2_score(y_train,trainPredict)
print("R2 Training Dataset Score", r2TrainingScore)

# Model Evaluation (Testing Data)
testPredict = model.predict(x_test)
r2TestScore = metrics.r2_score(y_test,testPredict)
print("R2 Testing Dataset Score", r2TestScore)

# Predictive Model
gender = int(input("Enter Gender (0 for male, 1 for female): "))
age = int(input("Enter Age: "))
height = float(input("Enter Height (in cm): "))
weight = float(input("Enter Weight (in kg): "))
duration = float(input("Enter Duration (in mins): "))
heart_rate = float(input("Enter Heart Rate: "))
body_temp = float(input("Enter Body Temperature: "))

input_data = {
    'Gender': gender,
    'Age': age,
    'Height': height,
    'Weight': weight,
    'Duration': duration,
    'Heart_Rate': heart_rate,
    'Body_Temp': body_temp
}

input_df = pd.DataFrame([input_data])
predicted_calories = model.predict(input_df)
print("Predicted Calories Burnt:", predicted_calories[0])

