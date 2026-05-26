from flask import Flask, render_template, request, jsonify
from recommender import (
    content_based_recommend,
    collaborative_recommend,
    search_by_year_and_genre,
    hybrid_recommend,
    get_all_genres,
    get_all_years,
    get_all_titles,
    df
)

app = Flask(__name__)


# ─────────────────────────────────────────────
# HOME ROUTE
# ─────────────────────────────────────────────
@app.route('/')
def index():
    genres = get_all_genres()
    years = get_all_years()
    titles = get_all_titles()
    return render_template('index.html', genres=genres, years=years, titles=titles)


# ─────────────────────────────────────────────
# API: CONTENT-BASED RECOMMENDATION
# ─────────────────────────────────────────────
@app.route('/recommend/content', methods=['POST'])
def recommend_content():
    data = request.json
    title = data.get('title', '')
    results = content_based_recommend(title, top_n=6)
    return jsonify({'results': results, 'method': 'Content-Based (Cosine Similarity)', 'query': title})


# ─────────────────────────────────────────────
# API: COLLABORATIVE FILTERING
# ─────────────────────────────────────────────
@app.route('/recommend/collaborative', methods=['POST'])
def recommend_collaborative():
    data = request.json
    user_id = int(data.get('user_id', 0))
    results = collaborative_recommend(user_id, top_n=6)
    return jsonify({'results': results, 'method': 'Collaborative Filtering', 'query': f'User {user_id}'})


# ─────────────────────────────────────────────
# API: SEARCH BY YEAR AND GENRE
# ─────────────────────────────────────────────
@app.route('/recommend/search', methods=['POST'])
def recommend_search():
    data = request.json
    year = data.get('year', None)
    genre = data.get('genre', None)
    results = search_by_year_and_genre(year=year, genre=genre, top_n=12)
    return jsonify({'results': results, 'method': 'Genre/Year Filter', 'query': f'{genre} | {year}'})


# ─────────────────────────────────────────────
# API: HYBRID RECOMMENDATION
# ─────────────────────────────────────────────
@app.route('/recommend/hybrid', methods=['POST'])
def recommend_hybrid():
    data = request.json
    title = data.get('title', None)
    genre = data.get('genre', None)
    year = data.get('year', None)
    user_id = int(data.get('user_id', 0))
    results = hybrid_recommend(title=title, genre=genre, year=year, user_id=user_id, top_n=6)
    return jsonify({'results': results, 'method': 'Hybrid Recommendation', 'query': title or genre or year})


# ─────────────────────────────────────────────
# API: GET MOVIE DETAILS
# ─────────────────────────────────────────────
@app.route('/movie/<int:movie_id>', methods=['GET'])
def movie_detail(movie_id):
    if movie_id < 0 or movie_id >= len(df):
        return jsonify({'error': 'Movie not found'}), 404
    movie = df.iloc[movie_id].to_dict()
    return jsonify(movie)


if __name__ == '__main__':
    print("🎬 Movie Recommender System Running!")
    print("📍 Open: http://127.0.0.1:5000")
    app.run(debug=True)