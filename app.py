from flask import Flask, request, jsonify
import mlflow.sklearn
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load movies data and features (assumes movies.csv is present)
movies = pd.read_csv('movies.csv')
movies['title'] = movies['title'].fillna('')
movies['genres'] = movies['genres'].fillna('')
movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').astype(float)
movies['clean_title'] = movies['title'].str.replace(r'\(\d{4}\)', '', regex=True).str.strip()
movies['combined_features'] = (movies['clean_title'] + ' ' + movies['genres']).str.lower()

# Load production model from MLflow
model = mlflow.sklearn.load_model("models:/Movie_Recommender_Model/Production")

# For simplicity, re-create the same TF-IDF and SVD pipeline as in training
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1,2), min_df=2)
X_tfidf = tfidf.fit_transform(movies['combined_features'])
svd = TruncatedSVD(n_components=100, random_state=42)
X_svd = svd.fit_transform(X_tfidf)

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    movie_name = data.get('movie_name', '')
    n = int(data.get('n', 5))
    idx = movies[movies['clean_title'].str.lower() == movie_name.lower()].index
    if len(idx) == 0:
        return jsonify({'error': 'Movie not found!'}), 404
    idx = idx[0]
    distances, indices = model.kneighbors([X_svd[idx]], n_neighbors=n+1)
    recs = movies.iloc[indices[0][1:]][['title', 'genres']]
    return jsonify({'recommendations': recs.to_dict(orient='records')})

if __name__ == '__main__':
    app.run(debug=True)
