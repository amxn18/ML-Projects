import pandas as pd
import numpy as np
import difflib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset and reset index if needed
df = pd.read_csv('movies.csv')
df.reset_index(inplace=True)  # Adds an 'index' column if not already present

# Features to combine for content
features = ['genres', 'keywords', 'tagline', 'cast', 'director']

# Fill NaN values with empty string
for feature in features:
    df[feature] = df[feature].fillna('')

# Combine all selected features into a single string
combinedFeatures = df['genres'] + ' ' + df['keywords'] + ' ' + df['tagline'] + ' ' + df['cast'] + ' ' + df['director']

# Vectorize the combined features
vectorizer = TfidfVectorizer()
vectorizedData = vectorizer.fit_transform(combinedFeatures)

# Compute cosine similarity between movies
similarity = cosine_similarity(vectorizedData)

# Ask user for a movie name
MovieName = input("Enter the movie name: ")

# List of all movie titles
titles = df['title'].tolist()

# Get close matches using difflib
closeMatch = difflib.get_close_matches(MovieName, titles, n=3, cutoff=0.5)

# If no close match found, show suggestions or fallback message
if not closeMatch:
    print("No matching movie found. Please check the spelling.")
    print("Here are some suggestions:\n")

    # Show top 5 titles that contain the input as a substring (case-insensitive)
    suggestions = [title for title in titles if MovieName.lower() in title.lower()]
    suggestions = suggestions[:5]  # Limit to 5

    if suggestions:
        for s in suggestions:
            print("- " + s)
    else:
        print("No suggestions available. Try using simpler or more accurate movie names.")

# If match is found, continue with recommendation
else:
    closestMatch = closeMatch[0]
    print("Closest match: ", closestMatch)

    # Find index of the matched movie
    idx = df[df.title == closestMatch]['index'].values[0]
    print("Index of the movie: ", idx)

    # Get similarity scores for the matched movie
    similarityScore = list(enumerate(similarity[idx]))
    sortedSimilarityScore = sorted(similarityScore, key=lambda x: x[1], reverse=True)

    # Print top 30 most similar movies
    print("\nMovies Suggested for you:\n")
    i = 1
    for movie in sortedSimilarityScore:
        index = movie[0]
        titleIdx = df[df.index == index]['title'].values[0]
        if i < 30:
            print(i, ":", titleIdx)
            i += 1
