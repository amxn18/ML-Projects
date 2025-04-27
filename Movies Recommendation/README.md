# 🎬 Movie Recommendation System
- **Dataset**: [Movies Dataset on Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)
```
A content-based movie recommender using TF-IDF Vectorization and Cosine Similarity
```

---

## 📌 Project Overview

This project recommends movies similar to a user-given title based on metadata like:
- **Genres**
- **Keywords**
- **Tagline**
- **Cast**
- **Director**

Instead of collaborative filtering, it uses **content-based filtering**, meaning it finds similarity between movies based on their descriptions and metadata, not user ratings.

---

## 🔧 How It Works

```
Step 1: Clean the data and fill missing values
Step 2: Combine key text features into a single string per movie
Step 3: Use TF-IDF Vectorizer to convert text into numeric vectors
Step 4: Compute cosine similarity between all movie vectors
Step 5: Take movie input from user and recommend top 30 similar ones
```

Also includes **typo handling**: if the user misspells a movie name, it suggests the closest match and continues smoothly.

---

## 🛠 Features

```
✔️ Content-based filtering using movie metadata
✔️ TF-IDF + Cosine Similarity for relevance scoring
✔️ Fuzzy matching with difflib for typos and partial names
✔️ Clean user interaction with fallback suggestions
```

---

## 💻 File Structure

```
movie-recommender/
│
├── movies.csv              # Dataset containing movie info
├── movie_recommender.py    # Main Python script
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 🧪 Example

```
> Enter the movie name: avatr

Closest match:  Avatar
Index of the movie: 102

Movies Suggested for you:

1 : Avatar
2 : Guardians of the Galaxy
3 : Star Trek
...
```

If input is unrecognized:
```
> Enter the movie name: zzzzzz

No matching movie found. Please check the spelling.
Here are some suggestions:

- Zookeeper
- Zodiac
- Zoom
```

---

## 📦 Dependencies

Install required libraries:

```bash
pip install -r requirements.txt
```

Packages used:
```
pandas
numpy
scikit-learn
difflib (built-in)
```

---

## 🚀 Run the Recommender

```bash
python movie_recommender.py
```

---

## 📚 Dataset Info

- File: `movies.csv`
- Must include columns: `title`, `genres`, `keywords`, `tagline`, `cast`, `director`
- If no `index` column is present, it's added using `df.reset_index()`

---

## 🙋‍♂️ Author

```
Aman Kukreja
GitHub: https://github.com/amxn18
LinkedIn: https://linkedin.com/in/amankukreja18
```

---

## ⭐ Extras (Optional Ideas)

- Add a retry loop for user input
- Build a Streamlit or Flask web app
- Expand with collaborative filtering or hybrid systems
