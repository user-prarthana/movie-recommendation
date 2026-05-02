import pandas as pd
import ast
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "data", "movies.csv")

df = pd.read_csv(csv_path)

# CLEAN GENRES
def clean_genres(text):
    try:
        if isinstance(text, str) and text.startswith("["):
            items = ast.literal_eval(text)
            return " ".join([i['name'].lower() for i in items])
        return str(text).lower()
    except:
        return ""

df['genres'] = df['genres'].fillna("").apply(clean_genres)
df['keywords'] = df['keywords'].fillna("").astype(str).str.lower()
df['overview'] = df['overview'].fillna("No description available")

# FEATURE
df['combined'] = df['genres'] + " " + df['keywords'] + " " + df['overview']

vectorizer = TfidfVectorizer(stop_words='english')
matrix = vectorizer.fit_transform(df['combined'])
similarity = cosine_similarity(matrix)

# POSTER FIX
def get_poster(path, title):
    if pd.isna(path) or path == "" or path == "null":
        return f"https://via.placeholder.com/300x450?text={title.replace(' ', '+')}"
    return "https://image.tmdb.org/t/p/w500" + path

# FIND MOVIE
def find_movie(title):
    title = title.lower()
    titles = df['title'].str.lower().tolist()

    if title in titles:
        return title

    match = get_close_matches(title, titles, n=1)
    return match[0] if match else None

# FORMAT
def format_row(row):
    return {
        "title": row['title'],
        "overview": row['overview'],
        "rating": float(row.get('vote_average', 0)),
        "poster": get_poster(row.get('poster_path'), row['title'])
    }

# AI RECOMMEND
def recommend(title):
    title = find_movie(title)
    if not title:
        return []

    idx = df[df['title'].str.lower() == title].index[0]
    scores = sorted(list(enumerate(similarity[idx])), key=lambda x: x[1], reverse=True)[1:9]

    return [format_row(df.iloc[i[0]]) for i in scores]

# GENRE
def recommend_by_selected_genre(genre):
    genre = genre.lower().strip()

    results = []
    for _, row in df.iterrows():
        if genre in row['genres']:
            results.append(format_row(row))

    return results[:8]