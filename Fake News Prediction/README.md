# Fake-News-Prediction
## 📥 Download Full Dataset
Download the full dataset from:
[Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
# 📰 Fake News Detection using Logistic Regression

This project is a simple **Fake News Detection** system that classifies news articles as **fake** or **real** using **Natural Language Processing (NLP)** and **Logistic Regression**.

---

## 📁 Dataset

The dataset used is `fakeNews.csv` which contains the following columns:

- `author`: Author of the news article
- `title`: Title of the news article
- `label`: 0 for Fake News, 1 for Real News

---

## 📌 Features

- Combines `author` and `title` to form the content
- Applies text preprocessing:
  - Lowercasing
  - Removing punctuation
  - Removing stopwords
  - Stemming
- Converts text into numerical features using **TF-IDF Vectorizer**
- Splits the data into training and testing sets
- Trains a **Logistic Regression** model
- Evaluates using **accuracy score**
- Predicts if a new news article is real or fake

---

## 🔧 Libraries Used

```bash
numpy
pandas
nltk
sklearn
re
