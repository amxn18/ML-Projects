import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score

diabitic = pd.read_csv('diabetes.csv')
# 0--> non-diabitic 1--> diabitic

#  Splitting the data into features and target
x = diabitic.drop(columns='Outcome', axis=1)
y = diabitic['Outcome'] 

# Standardizing the data
standard = StandardScaler()
a = standard.fit_transform(x)

# Splitting the data into training and test data
x_train, x_test, y_train, y_test = train_test_split(a, y, test_size=0.2, stratify=y, random_state=2)

# Support Vector Machine
model = svm.SVC(kernel='rbf')
model.fit(x_train, y_train)

# Accuracy score --> Training data
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)        
print('Accuracy score of training data:', training_data_accuracy)
          

# Accuracy score --> Testing data
x_test_prediction = model.predict(x_test)
testing_data_accuracy = accuracy_score(x_test_prediction, y_test)        
print('Accuracy score of testing data:', testing_data_accuracy)


# Prediction
inputData = (3,158,76,36,245,31.6,0.851,28)
inputDataArray = np.asarray(inputData)
# Will tell model that we need prediction for 1 data point
inputDataReshaped = inputDataArray.reshape(1, -1)

#  Standardizing the input data
standarInputData = standard.transform(inputDataReshaped)
prediction = model.predict(standarInputData)
if prediction[0] == 0:
    print('The person is not diabitic')
else:
    print('The person is diabitic')