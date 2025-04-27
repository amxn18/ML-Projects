import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

wineData = pd.read_csv("winequality-red.csv", delimiter=';')
# print(wineData.isnull().sum())

# Data analysis
# Visualisation
sns.catplot(x='quality', data= wineData, kind = 'count')
plt.title('Value Count')
# print(wineData['quality'].value_counts())

# 1) volatile acidity and quality (Inversely proportonal)
plot = plt.figure(figsize=(5,5))
plt.title('volatile acidity Vs quality')
sns.barplot(x='quality', y = 'volatile acidity', data = wineData)

# 2) citric acidity vs quality(Directly proportional)
plot = plt.figure(figsize=(5,5))
plt.title('citric acidity Vs quality')
sns.barplot(x='quality', y = 'citric acid', data = wineData)

# 3) residual sugar vs quality(Directly proportional)
plot = plt.figure(figsize=(5,5))
plt.title('residual sugar Vs quality')
sns.barplot(x='quality', y = 'residual sugar', data = wineData)

# Corelation 1) Positive correlation 2) Negative Correlation
correlation = wineData.corr()
# Heatmap to understand the correlation
plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot = True, annot_kws= {'size' : 8}, cmap = 'Blues')
# plt.show()

# Separating Features and labels 
x = wineData.drop('quality', axis=1)

# Label Binarization (>7 --> good, <6 --> Bad)
y = wineData['quality'].apply(lambda yVal: 1 if yVal>=7 else 0)


# Splitting the data into training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

# Training The model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Model Evaluation (Training Accuracy)
trainingPrediction = model.predict(x_train)
trainingScore = accuracy_score(trainingPrediction, y_train)
print("Accuracy of Training Data:",trainingScore) 

# Model Evaluation (Test Accuracy)
testPrediction = model.predict(x_test)
testScore = accuracy_score(testPrediction, y_test)
print("Accuracy of Test Data:",testScore) 
# Accuracy ~ 0.93 which means out of 100 datapoints our model can predict 93 datapoints correctly

# Predictive Model
inputData = (7.4,0.7,0,1.9,0.076,11,34,0.9978,3.51,0.56,9.4)
inputDataArray = np.asarray(inputData)
reshapedData =inputDataArray.reshape(1,-1)
prediction = model.predict(reshapedData)
# print(prediction)

if(prediction[0] == 1): print("Wine is of Good quality")
else: print("Wine is of bad quality")


# Copy any row of values from dataset and past it in inputData make sure to remove semcolan and add commas instead

