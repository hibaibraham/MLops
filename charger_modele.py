
import mlflow
import mlflow.sklearn  # pour recharger un modèle scikit-learn
import pandas as pd


# ===== 1. Retrouver le meilleur run =====
# Rechercher les runs de l'expérience et les trier par score F1 décroissant
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    order_by=["metrics.f1 DESC"]
)

run_id = runs.iloc[0]["run_id"]  # identifiant du meilleur run


# ===== 2. Recharger le modèle =====
# "runs:/<id>/model" désigne le modèle sauvegardé dans ce run
model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

print("Modele recharge :", type(model).__name__)
print("Nombre de variables attendues :", model.n_features_in_)


# ===== 3. Créer un nouveau client =====
# Les colonnes d'abonnement sont transformées en valeurs True/False
# Ici : abonnement Standard (Premium = False)
client = pd.DataFrame([{
    "age": 35,
    "salaire": 2500,
    "anciennete": 1,
    "consommation": 10,
    "abonnement_Premium": False,
    "abonnement_Standard": True
}])


# ===== 4. Garder les colonnes attendues par le modèle =====
# Respecter les noms et l'ordre des variables utilisées pendant l'entraînement
# Selon le run choisi, la colonne salaire peut être utilisée ou non
client = client[list(model.feature_names_in_)]


# ===== 5. Prédire si le client va partir =====
# 1 = le client va partir, 0 = le client reste
print(
    "Le client va-t-il partir ? ->",
    "OUI" if model.predict(client)[0] == 1 else "NON"
)
