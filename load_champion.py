import mlflow
import mlflow.sklearn
import pandas as pd

# 1. Connexion à la base de données de suivi MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# 2. Chargement du modèle "champion" directement via son alias
model_uri = "models:/churn-model@champion"
model = mlflow.sklearn.load_model(model_uri)

print("Modèle chargé :", type(model).__name__)
print("Variables attendues par le modèle :", list(model.feature_names_in_))

# 3. Définition des caractéristiques du nouveau client à tester
client = pd.DataFrame([{
    "age": 35,
    "salaire": 2500,
    "anciennete": 1,
    "consommation": 10,
    "abonnement_Premium": False,
    "abonnement_Standard": True,
}])

# 4. Alignement strict des colonnes sur celles attendues par le modèle
client = client[list(model.feature_names_in_)]

# 5. Prédiction du churn (1 = départ, 0 = fidélité)
prediction = model.predict(client)[0]

# 6. Affichage du résultat
decision = "OUI" if prediction == 1 else "NON"
print(f"Le client va-t-il partir ? -> {decision}")