import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn. linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

dataT = pd.read_csv('train.csv')
# print(dataT.head())
# print(dataT.isnull().sum())
# print(dataT.describe())

# Handelling Missing Values 
# 1) Age
# 2) Cabin No(No need so we will drop this column)
# 3) Embarked 

dataT = dataT.drop(columns='Cabin', axis = 1)
# print(dataT.isnull().sum())

# Handelling Missing Age Values
dataT['Age'].fillna(dataT['Age'].mean(), inplace=True)
# print(dataT.isnull().sum())

# Handelling Missing Embarked Values
dataT['Embarked'].fillna(dataT['Embarked'].mode()[0], inplace= True)
# print(dataT.isnull().sum())

# Data Ananlysis
# print(dataT['Survived'].value_counts()) # 0 = No, 1 = Yes
# print(dataT['Sex'].value_counts())

# Data Visualisation
sns.countplot(x='Survived', data = dataT)
# plt.show()
sns.countplot(x='Sex', hue='Survived',data = dataT)
# plt.show()

#  Encoding Categorical Columns
dataT.replace({
    'Sex': {'male': 0, 'female': 1},
    'Embarked': {'S': 0, 'C': 1, 'Q': 2}
}, inplace=True)

# print(dataT.head())

# Separating Features and Target
x = dataT.drop(columns=['PassengerId', 'Name', 'Ticket', 'Survived'], axis = 1)
y = dataT['Survived']
# print(x.head())

# Splitting the data into training and test dataset
x_train , x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=2)

# Model 1 --> LogisticRegression
model1 = LogisticRegression()
model1.fit(x_train, y_train)

# Model 1 Accuracy Score (Training Dataset)
trainPredict1 = model1.predict(x_train)
trainScore1 = accuracy_score(y_train, trainPredict1)
print("Logitic Regression Training Accuracy Score:", trainScore1)

# Model 1 Accuracy Score (Testing Dataset)
testPredict1 = model1.predict(x_test)
testScore1 = accuracy_score(y_test, testPredict1)
print("Logitic Regression Testing Dataset  Accuracy Score:", testScore1)

# Model 2 --> XGBClassifier
model2 = XGBClassifier()
model2.fit(x_train, y_train)

# Model 2 Accuracy Score (Training Dataset)
trainPredict2 = model2.predict(x_train)
trainScore2 = accuracy_score(y_train, trainPredict2)
print("XGBClassifier Training Accuracy Score:", trainScore2)

# Model 2 Accuracy Score (Testing Dataset)
testPredict2 = model2.predict(x_test)
testScore2 = accuracy_score(y_test, testPredict2)
print("XGBClassifier Testing Dataset  Accuracy Score:", testScore2)

# Predictive Model
# ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
# Predictive System using XGBoost (you can also use model1 for Logistic Regression)

input_data = ()  # Example input (Pclass, Sex, Age, SibSp, Parch, Fare, Embarked)

# Convert input data to numpy array and reshape
input_array = np.asarray(input_data).reshape(1, -1)

# Make prediction
prediction = model2.predict(input_array)

# Show result
if prediction[0] == 0:
    print(" Passenger did not survive.")
else:
    print(" Passenger survived.")

# Add values in particular order from dataset to check
