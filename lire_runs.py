
import mlflow  # réutiliser MLflow pour lire les résultats

# Récupérer tous les runs de l'expérience et les trier par F1 décroissant
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    order_by=["metrics.f1 DESC"]
)

# Colonnes à afficher : nom du run, paramètres et métriques
colonnes = [
    "tags.mlflow.runName",
    "params.n_estimators",
    "params.max_depth",
    "metrics.accuracy",
    "metrics.f1"
]

print(runs[colonnes].to_string(index=False))  # afficher sans numéros de ligne

# La première ligne correspond au run ayant le meilleur score F1
meilleur = runs.iloc[0]

print()

# Afficher le nom du meilleur run et les 8 premiers caractères de son identifiant
print(
    "Meilleur run :",
    meilleur["tags.mlflow.runName"],
    "| id :",
    meilleur["run_id"][:8],
    "..."
)
