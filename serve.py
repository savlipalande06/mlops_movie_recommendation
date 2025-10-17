from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import mlflow.pyfunc
import os

# ======================================
# Configuration
# ======================================

# The folder structure will be: ./exported_model
MODEL_URI_LOCAL = "./exported_model" 
MOVIES_DATA_PATH = "movies.csv"


# ======================================
# Initialization and Model Loading
# ======================================

app = FastAPI(title="MLflow Movie Recommender API")

# Global variables to be populated at startup
model = None
movies_df = None

class RecommendationRequest(BaseModel):
    movie_name: str = Field(..., example="Toy Story (1995)")
    n_recommendations: int = Field(5, example=5)

def load_data_and_model():
    """Load the model and data into memory on application startup."""
    global model, movies_df
    
    # 1. Load Data
    try:
        movies_df = pd.read_csv(MOVIES_DATA_PATH)
        # Replicate the essential preprocessing done in the training script
        movies_df['title'] = movies_df['title'].fillna('')
        movies_df['genres'] = movies_df['genres'].fillna('')
        movies_df['clean_title'] = movies_df['title'].str.replace(r'\s*\(\d{4}\)', '', regex=True).str.strip()
        movies_df['combined_features'] = (movies_df['clean_title'] + ' ' + movies_df['genres']).str.lower()
        movies_df = movies_df.reset_index()
        print(f"✅ Movie data loaded successfully! Shape: {movies_df.shape}")
    except FileNotFoundError:
        print(f"❌ Error: {MOVIES_DATA_PATH} not found.")
        movies_df = None
        
    # 2. Load Model
    try:
        # Load the MLflow PyFunc model (the SKLearn Pipeline)
        # This requires the tracking server to be running or a local copy of the model to exist.
        model = mlflow.pyfunc.load_model(MODEL_URI)
        print(f"✅ MLflow model loaded successfully from URI: {MODEL_URI}")
    except Exception as e:
        print(f"❌ Failed to load MLflow model from Registry: {e}")
        print("Ensure MLflow Tracking Server is accessible and the model alias/name is correct.")
        model = None

# Execute model and data loading on startup
load_data_and_model()


# ======================================
# Helper/Prediction Logic
# ======================================

def get_recommendations_inference(movie_name: str, n: int = 5):
    """Generates movie recommendations using the loaded MLflow model."""
    if model is None or movies_df is None:
        raise HTTPException(status_code=500, detail="Recommendation model or data not loaded.")

     # The Custom PyFunc expects a DataFrame input matching the required schema
    input_df = pd.DataFrame({
        'movie_name_input': [movie_name], 
        'n_recommendations_input': [n]
    })
    


    try:
        # Assume the PyFunc model's predict method handles the full recommendation logic
        # and returns a list of recommended movie titles.
        recommended_movies_list = model.predict(X_input_df)
        
       if len(recommended_movies_list) == 1 and recommended_movies_list[0].startswith("Error:"):
            return {"error": recommended_movies_list[0]}
             
        return {
            "movie_name": movie_name,
            "recommendations": recommended_movies_list,
            "model_path": MODEL_URI_LOCAL
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error during inference: {e}")


# ======================================
# API Endpoints
# ======================================

class RecommendationRequest(BaseModel):
    # Input schema for the movie title
    movie_name: str = Field(..., example="Toy Story (1995)", description="The title of the movie to get recommendations for.")
    n_recommendations: int = Field(5, example=5, description="Number of recommendations to return.")


@app.get("/")
def root():
    """Health check endpoint."""
    if model is not None:
        return {"message": "✅ MLflow Movie Recommender API is live and ready."}
    else:
        # If model loading failed at startup
        raise HTTPException(status_code=503, detail="❌ MLflow Model not loaded. Service unavailable.")

@app.post("/recommend")
def recommend_api(request: RecommendationRequest):
    """Generates the top N movie recommendations."""
    
    # Use the helper function to get recommendations
    results = get_recommendations_inference(request.movie_name, request.n_recommendations)
    
    # Check if the helper returned a 'Movie not found' error
    if "error" in results:
        # Return 404 for resource not found
        raise HTTPException(status_code=404, detail=results["error"])
        
    return results