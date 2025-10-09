# 🎬 Personalized Movie Recommendation System with Full MLOps Lifecycle

# ======================================
# 0. Setup (Run on Colab)
# ======================================
#!pip install mlflow scikit-learn seaborn matplotlib pandas

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
import os
import hashlib

# ======================================
# 1. Load Dataset (Run on Colab)
# ======================================
#from google.colab import files
#uploaded = files.upload()  # upload movies.csv


# --- Data Versioning: Compute hash of movies.csv ---
def file_hash(filepath):
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

movies_csv_path = 'movies.csv'
movies = pd.read_csv(movies_csv_path)
data_hash = file_hash(movies_csv_path)
print(f"Dataset shape: {movies.shape}")
print(movies.head())
print(f"Data hash: {data_hash}")

# ======================================
# 2. Preprocess Titles and Genres
# ======================================
movies['title'] = movies['title'].fillna('')
movies['genres'] = movies['genres'].fillna('')
movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').astype(float)
movies['clean_title'] = movies['title'].str.replace(r'\(\d{4}\)', '', regex=True).str.strip()
movies['combined_features'] = (movies['clean_title'] + ' ' + movies['genres']).str.lower()

# ======================================
# 3. TF-IDF Feature Extraction
# ======================================
tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1,2), min_df=2)
X_tfidf = tfidf.fit_transform(movies['combined_features'])
print(f"TF-IDF matrix shape: {X_tfidf.shape}")

# ======================================
# 4. Dimensionality Reduction (SVD)
# ======================================
svd = TruncatedSVD(n_components=100, random_state=42)
X_svd = svd.fit_transform(X_tfidf)

# ======================================
# 5. Model Training (KNN)
# ======================================
knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(X_svd)

# ======================================
# 6. Representative Movies
# ======================================
genre_counts = movies['genres'].str.split('|').explode().value_counts().to_dict()

def representativeness(genres):
    if pd.isna(genres):
        return 0
    parts = genres.split('|')
    return np.mean([genre_counts.get(p, 0) for p in parts])

movies['rep_score'] = movies['genres'].apply(representativeness)
top5 = movies.sort_values('rep_score', ascending=False).head(5)
print("\nTop 5 Representative Movies:")
print(top5[['movieId', 'title', 'genres', 'rep_score']])

# ======================================
# 7. Recommendation Function
# ======================================
def recommend(movie_name, n=5):
    idx = movies[movies['clean_title'].str.lower() == movie_name.lower()].index
    if len(idx) == 0:
        print("Movie not found!")
        return []
    idx = idx[0]
    distances, indices = knn.kneighbors([X_svd[idx]], n_neighbors=n+1)
    recs = movies.iloc[indices[0][1:]][['title', 'genres']]
    print(f"\nMovies similar to '{movie_name}':")
    print(recs)
    return recs

# recommend('Toy Story')

# ======================================
# 8. MLflow Setup (Colab + Docker MLflow)
# ======================================
# 👉 Run MLflow server locally (Docker):
# docker build -t movie-recommender-mlflow .
# docker run -p 5000:5000 movie-recommender-mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Docker MLflow instance
mlflow.set_experiment("Movie_Recommendation_System")

# ======================================
# 9. Start Run, Log, and Register Model
# ======================================

with mlflow.start_run(run_name="Movie_Rec_TFIDF_SVD_KNN") as run:
    mlflow.log_param("vectorizer", "TF-IDF")
    mlflow.log_param("dim_reduction", "SVD (100 comps)")
    mlflow.log_param("model", "KNN (cosine)")
    mlflow.log_param("data_hash", data_hash)
    mlflow.log_metric("n_movies", len(movies))
    mlflow.log_metric("n_features", X_tfidf.shape[1])

    # Log the raw data file as an artifact for full data versioning
    mlflow.log_artifact(movies_csv_path)

    mlflow.sklearn.log_model(knn, "knn_recommender_model")

    top5.to_csv('top5_movies.csv', index=False)
    mlflow.log_artifact('top5_movies.csv')

    model_uri = f"runs:/{run.info.run_id}/knn_recommender_model"
    model_details = mlflow.register_model(model_uri, "Movie_Recommender_Model")

print("✅ Model registered successfully in MLflow.")

# ======================================
# 10. Transition to Production
# ======================================
from mlflow.tracking import MlflowClient
client = MlflowClient()

client.transition_model_version_stage(
    name="Movie_Recommender_Model",
    version=model_details.version,
    stage="Production",
    archive_existing_versions=True
)
print("✅ Model transitioned to Production stage.")

# ======================================
# 11. Load Production Model
# ======================================
loaded_model = mlflow.sklearn.load_model("models:/Movie_Recommender_Model/Production")
print("Loaded production model successfully!")

# ======================================
# 12. Extra Insights (Visualization)
# ======================================
plt.figure(figsize=(10,5))
top_genres = pd.Series(genre_counts).head(10)
sns.barplot(x=top_genres.values, y=top_genres.index)
plt.title('Top 10 Movie Genres')
plt.xlabel('Count')
plt.ylabel('Genre')
plt.tight_layout()
plt.savefig('genre_distribution.png')
mlflow.log_artifact('genre_distribution.png')
plt.show()

# ======================================
# 13. Version Control (Git - run locally)
# ======================================
# git init
# git add .
# git commit -m "Initial Movie Recommender MLOps setup"
# git remote add origin https://github.com/<your-username>/movie-recommender-mlops.git
# git push -u origin main
# git tag -a v1.0 -m "First production recommender model"
# git push origin v1.0

# ======================================
# 14. Docker Commands (run locally)
# ======================================
# docker build -t movie-recommender-mlflow .
# docker run -p 5000:5000 movie-recommender-mlflow


# ======================================
# 15. Flask API for Model Serving
# ======================================
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/recommend', methods=['POST'])
def recommend_api():
    data = request.get_json()
    movie_name = data.get('movie_name', '')
    n = int(data.get('n', 5))
    idx = movies[movies['clean_title'].str.lower() == movie_name.lower()].index
    if len(idx) == 0:
        return jsonify({'error': 'Movie not found!'}), 404
    idx = idx[0]
    distances, indices = loaded_model.kneighbors([X_svd[idx]], n_neighbors=n+1)
    recs = movies.iloc[indices[0][1:]][['title', 'genres']]
    return jsonify({'recommendations': recs.to_dict(orient='records')})

if __name__ == '__main__':
    app.run(debug=True)

print("\n🎯 Complete MLOps lifecycle executed successfully!")

