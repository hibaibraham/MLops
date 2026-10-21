
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ===== 1. Charger et préparer les données =====
df = pd.read_csv("churn_data.csv")

X = pd.get_dummies(
    df[["age", "salaire", "abonnement", "anciennete", "consommation"]],
    drop_first=True
)

# Diviser les données en 80 % pour l'entraînement et 20 % pour le test
X_train, X_test, y_train, y_test = train_test_split(
    X, df["churn"], test_size=0.2, random_state=42
)

# ===== 2. Configurer l'expérience MLflow =====
mlflow.set_experiment("churn-prediction")

# Activer l'enregistrement automatique pour scikit-learn :
# MLflow enregistre automatiquement les paramètres, les métriques
# et le modèle entraîné.
mlflow.sklearn.autolog(serialization_format="cloudpickle")

# ===== 3. Entraîner le modèle =====
with mlflow.start_run(run_name="essai-auto"):

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=4,
        random_state=42
    )

    model.fit(X_train, y_train)  # l'enregistrement se déclenche au fit()

    # Calculer l'accuracy sur les données de test
    print("score test :", model.score(X_test, y_test))
