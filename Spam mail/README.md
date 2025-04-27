# 📧 Spam Mail Detection using Logistic Regression

This is a Machine Learning project that detects whether a given email/message is **SPAM** or **NOT SPAM** using a Logistic Regression classifier. The dataset used is a labeled collection of real-world email messages.

──────────────────────────────────────────────

# 📂 Dataset
The dataset `mail_data.csv` contains two columns:
- `Category` — Label (spam or ham)
- `Message` — The email or text message content

──────────────────────────────────────────────

# 🔧 Libraries Used

- pandas
- numpy
- scikit-learn (sklearn)

──────────────────────────────────────────────

# 🧠 ML Workflow

1. **Data Preprocessing**
   - Null value handling
   - Encoding labels (spam → 1, ham → 0)

2. **Feature Extraction**
   - Used `TfidfVectorizer` for converting text into numerical vectors
   - Removed English stopwords and applied lowercase normalization

3. **Model Training**
   - Split the dataset into training and testing sets (80-20 split)
   - Trained a **Logistic Regression** model on the training set

4. **Model Evaluation**
   - Used `accuracy_score` to evaluate model performance on both training and testing sets

5. **Prediction System**
   - A sample message is transformed and passed to the model for classification

──────────────────────────────────────────────

# 🧪 Accuracy Results

- ✅ Training Accuracy: ~97%
- ✅ Testing Accuracy: ~96%

──────────────────────────────────────────────

# 📌 How to Use

1. Clone the repository

```bash
git clone https://github.com/yourusername/spam-mail-detection.git
cd spam-mail-detection
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the project

```bash
python spam_detector.py
```

──────────────────────────────────────────────

# 📥 Sample Prediction Code

```python
input_mail = ["Congratulations! You've won a free ticket. Call now!"]
input_data_vectorized = vectorizer.transform(input_mail)
prediction = model.predict(input_data_vectorized)

if prediction[0] == 1:
    print("The mail is SPAM")
else:
    print("The mail is NOT SPAM")
```

──────────────────────────────────────────────

# 👨‍💻 Author

- Name: Aman Kukreja
- GitHub: [amxn18](https://github.com/amxn18)
- LinkedIn: [Aman Kukreja](https://www.linkedin.com/in/amankukreja18/)

──────────────────────────────────────────────

# ⭐️ Give a Star!

If you liked this project, consider giving it a ⭐️ on GitHub!
```
