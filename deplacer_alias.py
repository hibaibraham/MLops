import mlflow
from mlflow import MlflowClient

# 1. Connexion à la base de données de suivi MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

# 2. Retrouver le run nommé « essai-2 »
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    filter_string="tags.mlflow.runName = 'essai-2'"
)
run_id = runs.iloc[0]["run_id"]

# 3. Enregistrer ce modèle (crée une NOUVELLE VERSION dans churn-model)
v = mlflow.register_model(f"runs:/{run_id}/model", "churn-model")
print(f"Nouvelle version créée : {v.version}")

# 4. Identifier quelle version détient l'alias « champion » actuellement
actuel = client.get_model_version_by_alias("churn-model", "champion")
print(f"Champion actuel : version {actuel.version}")

# 5. Déplacer temporairement l'alias vers la nouvelle version créée
client.set_registered_model_alias("churn-model", "champion", v.version)
v_apres = client.get_model_version_by_alias("churn-model", "champion").version
print(f"Après déplacement : champion -> version {v_apres}")

# 6. Réattribuer l'alias à la version championne d'origine
client.set_registered_model_alias("churn-model", "champion", actuel.version)
v_retour = client.get_model_version_by_alias("churn-model", "champion").version
print(f"Après retour : champion -> version {v_retour}")