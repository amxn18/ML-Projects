import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# nltk.download('stopwords')
train = pd.read_csv('fakeNews.csv')
# print(train.head())
# print(train.isnull().sum())
train = train.fillna('')
train['content'] = train['author']+' '+train['title']
x = train['content']
y = train['label']
stemmer = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [stemmer.stem(word) for word in stemmed_content if not word in set(stopwords.words('english'))]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content
train['content'] = train['content'].apply(stemming)
a = train['content'].values
b = train['label'].values
tfidf = TfidfVectorizer()
v = tfidf.fit_transform(a)
# print(v)

# Splitting the dataset into training and testing data
x_train, x_test, y_train, y_test = train_test_split(v, b, test_size=0.2, random_state=0)

# Training the model
model = LogisticRegression()
model.fit(x_train, y_train)

# Accuracy(Training)
trainingPrediction = model.predict(x_train)
traingAccuracy = accuracy_score(trainingPrediction, y_train)
print("Training Accuracy: ", traingAccuracy)

# Accuracy(Testing)
testingPrediction = model.predict(x_test)
testingAccuracy = accuracy_score(testingPrediction, y_test)
print("Testing Accuracy: ", testingAccuracy)

# Prediction

# Check News From Dataset
newNews = x_test[0]

# Check News From User

# inputNews = input("Enter News: ")
# processedNews = stemming(inputNews)
# newNews = tfidf.transform([processedNews])

# Predict
predict = model.predict(newNews)
print("Prediction: ", predict)

if predict == 0:
    print("Fake News")
else:
    print("Real News")



