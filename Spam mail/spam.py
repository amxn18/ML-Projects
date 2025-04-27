import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv("mail_data.csv")
# print(df.head())
# print(df.info())

mailData = df.where(pd.notnull(df), '')
# print(mailData.head())
# print(mailData.info())

encoder = LabelEncoder()
mailData['Category'] = encoder.fit_transform(mailData['Category'])  # 1--> spam, 0 --> ham
# print(mailData.head())

x = mailData['Message']
y = mailData['Category']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)
# print(x_train.shape)

# Feature Extraction
vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
x_train_vectorized = vectorizer.fit_transform(x_train)
x_test_vectorized = vectorizer.transform(x_test)

model = LogisticRegression()
model.fit(x_train_vectorized, y_train)

# Evalaution of training model
y_train_predicted = model.predict(x_train_vectorized)
training_data_accuracy = accuracy_score(y_train, y_train_predicted)
print("Training data accuracy: ", training_data_accuracy)

# Evalaution of testing model   
y_test_predicted = model.predict(x_test_vectorized)
testing_data_accuracy = accuracy_score(y_test, y_test_predicted)
print("Testing data accuracy: ", testing_data_accuracy)


# Input and Predictive Model

# Sample input message
input_mail = ["Congratulations! You've won a free ticket to Bahamas. Call now to claim."]

# Convert text to feature vector (same transformation used for training)
input_data_vectorized = vectorizer.transform(input_mail)

# Make prediction
prediction = model.predict(input_data_vectorized)

# Decode result
if prediction[0] == 1:
    print("The mail is SPAM")
else:
    print("The mail is NOT SPAM")


# Enter your own message to test the model