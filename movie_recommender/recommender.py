import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────
# LOAD & PREPARE DATA
# ─────────────────────────────────────────────
df = pd.read_csv('TMDB Movie Dataset.csv')

# Rename columns to match app expectations
df = df.rename(columns={
    'overview': 'description',
    'vote_average': 'rating',
    'genre': 'genres'
})

# Extract year from release_date (format: YYYY-MM-DD)
df['year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year.fillna(0).astype(int)

# Fill missing values
df['description'] = df['description'].fillna('')
df['genres'] = df['genres'].fillna('')
df['rating'] = df['rating'].fillna(0)
df['title'] = df['title'].fillna('Unknown')

# Clean genres: remove brackets and quotes
df['genres'] = df['genres'].str.replace(r"[\[\]']", '', regex=True).str.strip()

# Build combined feature string for TF-IDF
df['features'] = (
    df['genres'].str.replace(',', ' ', regex=False) + ' ' +
    df['description'] + ' ' +
    df['year'].astype(str)
)

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['features'])

# Cosine Similarity Matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Index map: title -> dataframe index
indices = pd.Series(df.index, index=df['title']).drop_duplicates()


# ─────────────────────────────────────────────
# SIMULATED USER-ITEM MATRIX (Collaborative Filtering)
# ─────────────────────────────────────────────
np.random.seed(42)
num_users = 20
num_movies = len(df)

user_item_matrix = np.zeros((num_users, num_movies))
for u in range(num_users):
    rated_movies = np.random.choice(num_movies, size=np.random.randint(10, 30), replace=False)
    for m in rated_movies:
        user_item_matrix[u, m] = np.random.randint(1, 6)

user_similarity = cosine_similarity(user_item_matrix)


# ─────────────────────────────────────────────
# 1. CONTENT-BASED RECOMMENDATION
# ─────────────────────────────────────────────
def content_based_recommend(title, top_n=6):
    if title not in indices:
        return []
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1]
    movie_indices = [i[0] for i in sim_scores]
    result = df.iloc[movie_indices][['title', 'genres', 'year', 'rating', 'description']].copy()
    result['score'] = [round(s[1] * 100, 1) for s in sim_scores]
    result['method'] = 'Content-Based (Cosine Similarity)'
    return result.to_dict('records')


# ─────────────────────────────────────────────
# 2. COLLABORATIVE FILTERING
# ─────────────────────────────────────────────
def collaborative_recommend(user_id=0, top_n=6):
    user_sims = user_similarity[user_id]
    similar_users = np.argsort(user_sims)[::-1][1:6]

    weighted_ratings = np.zeros(num_movies)
    sim_sum = np.zeros(num_movies)

    for similar_user in similar_users:
        sim = user_sims[similar_user]
        ratings = user_item_matrix[similar_user]
        weighted_ratings += sim * ratings
        sim_sum += sim * (ratings > 0).astype(float)

    predicted_ratings = np.where(sim_sum > 0, weighted_ratings / sim_sum, 0)
    unrated = np.where(user_item_matrix[user_id] == 0)[0]
    unrated_scores = [(i, predicted_ratings[i]) for i in unrated if predicted_ratings[i] > 0]
    unrated_scores = sorted(unrated_scores, key=lambda x: x[1], reverse=True)[:top_n]

    result = []
    for idx, score in unrated_scores:
        row = df.iloc[idx][['title', 'genres', 'year', 'rating', 'description']].to_dict()
        row['score'] = round(score * 20, 1)
        row['method'] = 'Collaborative Filtering'
        result.append(row)
    return result


# ─────────────────────────────────────────────
# 3. SEARCH BY YEAR AND GENRE
# ─────────────────────────────────────────────
def search_by_year_and_genre(year=None, genre=None, top_n=12):
    filtered = df.copy()

    if year:
     year = int(year)
    filtered = filtered[filtered['year'] == year]

    if genre and genre.lower() != 'all':
        filtered = filtered[filtered['genres'].str.contains(genre, case=False, na=False)]

    filtered = filtered.sort_values('rating', ascending=False).head(top_n)
    result = filtered[['title', 'genres', 'year', 'rating', 'description']].copy()
    result['score'] = result['rating'] * 10
    result['method'] = 'Genre/Year Filter'
    return result.to_dict('records')


# ─────────────────────────────────────────────
# 4. HYBRID RECOMMENDATION
# ─────────────────────────────────────────────
def hybrid_recommend(title=None, genre=None, year=None, user_id=0, top_n=6):
    results = []
    seen_titles = set()

    if title:
        for r in content_based_recommend(title, top_n=top_n):
            if r['title'] not in seen_titles:
                results.append(r)
                seen_titles.add(r['title'])

    for r in collaborative_recommend(user_id, top_n=top_n):
        if r['title'] not in seen_titles:
            results.append(r)
            seen_titles.add(r['title'])

    if genre or year:
        for r in search_by_year_and_genre(year=year, genre=genre, top_n=top_n):
            if r['title'] not in seen_titles:
                results.append(r)
                seen_titles.add(r['title'])

    return results[:top_n * 2]


# ─────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────
def get_all_genres():
    genres = set()
    for g in df['genres'].dropna():
        for item in g.split(','):
            clean = item.strip().strip("'\"")
            if clean:
                genres.add(clean)
    return sorted(genres)

def get_all_years():
    return sorted(df['year'][df['year'] > 0].unique().tolist(), reverse=True)

def get_all_titles():
    return sorted(df['title'].dropna().tolist())