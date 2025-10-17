# export_best_model.py
import os
import mlflow
from mlflow.tracking import MlflowClient
import joblib
import sys

EXPERIMENT_NAME = "movie_recommender_mlflow"  # must match your notebook

def main(out_path="C:\\Users\\Savli\\Downloads\\mlops_casestudy_Final\\movie_recommender_mlflow.py"):
    client = MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        print(f"ERROR: Experiment '{EXPERIMENT_NAME}' not found. Run the notebook first to create MLflow runs.")
        sys.exit(1)

    # search runs sorted by accuracy (descending)
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], order_by=["metrics.accuracy DESC"])
    if runs.shape[0] == 0:
        print("No runs found in experiment. Run the notebook first.")
        sys.exit(1)

    best = runs.iloc[0]
    run_id = best["run_id"]
    best_acc = best.get("metrics.accuracy", None)
    print(f"Best run_id = {run_id}  (accuracy={best_acc})")

    model_uri = f"runs:/{run_id}/model"
    print("Loading model from MLflow model uri:", model_uri)
    try:
        model = mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        print("Failed to load model from MLflow:", e)
        sys.exit(1)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(model, out_path)
    print(f"Saved best model to {out_path}")

if __name__ == "__main__":
    main()
