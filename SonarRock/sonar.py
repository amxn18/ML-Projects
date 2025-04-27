import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sonar = pd.read_csv('sonarData.csv', header=None)
encoder = LabelEncoder()
sonar[60] = encoder.fit_transform(sonar[60])

# Seprating the features and target
x = sonar.drop(columns=60, axis=1)
y = sonar[60]

# Seprating the dataset into train data and test data 
# 10% of the data will be used for testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, stratify=y, random_state=1)

# Training the logistic regression model with training data
model = LogisticRegression()
model.fit(x_train, y_train)

# Accuracy on training data
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)
print('Accuracy on training data : ', training_data_accuracy)

# Accuracy on test data
x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)  
print('Accuracy on test data : ', test_data_accuracy)

# Making a predictive system
inputData = ()
inputDataArray = np.asarray(inputData)

# Reshape the array as we are predicting for one instance
inputDataReshaped = inputDataArray.reshape(1, -1)

prediction = model.predict(inputDataReshaped)
if prediction[0] == 1:
    print('The object is a Rock')
else:
    print('The object is a Mine')

#  CHOOSE ANY ROW FROM THE sonarData.csv FILE AND PASTE THE VALUES IN THE inputData VARIABLE TO PREDICT WHETHER IT IS A MINE OR A ROCK