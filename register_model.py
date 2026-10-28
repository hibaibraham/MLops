import mlflow
from mlflow import MlflowClient

# 1. Connexion à la base de données de suivi MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# 2. Recherche du meilleur run selon le score F1 (ordre décroissant)
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    order_by=["metrics.f1 DESC"]
)

best_run = runs.iloc[0]
run_id = best_run["run_id"]
run_name = best_run.get("tags.mlflow.runName", "Sans nom")

print(f"Meilleur run : {run_name} | ID : {run_id[:8]}...")

# 3. Enregistrement du modèle dans le Registry
model_uri = f"runs:/{run_id}/model"
model_name = "churn-model"

result = mlflow.register_model(model_uri, model_name)
print(f"Modèle enregistré : {model_name} | Version : {result.version}")

# 4. Attribution de l'alias 'champion' à cette version
client = MlflowClient()
client.set_registered_model_alias(
    name=model_name,
    alias="champion",
    version=result.version
)
print(f"Alias 'champion' -> Version {result.version}")