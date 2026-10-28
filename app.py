from typing import Literal
from fastapi import FastAPI
import mlflow
import mlflow.sklearn
import pandas as pd
from pydantic import BaseModel

# 1. Création de l'application FastAPI
app = FastAPI(title="API Churn")

# 2. Connexion MLflow et chargement du modèle "champion" au démarrage
mlflow.set_tracking_uri("sqlite:///mlflow.db")
model = mlflow.sklearn.load_model("models:/churn-model@champion")


# 3. Schéma de données d'entrée avec Pydantic
class Client(BaseModel):
    age: int
    salaire: float
    abonnement: Literal["Basique", "Standard", "Premium"]
    anciennete: float
    consommation: float


# 4. Route de santé (Healthcheck)
@app.get("/health")
def health():
    return {"status": "ok"}


# 5. Route de prédiction
@app.post("/predict")
def predict(client: Client):
    # Transformation des données d'entrée
    ligne = pd.DataFrame([client.model_dump()])
    ligne = pd.get_dummies(ligne)

    # Réalignement dynamique des colonnes par rapport aux features attendues par le modèle
    ligne = ligne.reindex(columns=model.feature_names_in_, fill_value=0)

    # Calcul de la prédiction
    resultat = int(model.predict(ligne)[0])

    return {
        "churn": resultat,
        "message": "Le client va partir" if resultat == 1 else "Le client reste",
    }