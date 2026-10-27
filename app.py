
from typing import Literal

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 1. Créer l'application
app = FastAPI(title="API Churn")

# 2. Configurer MLflow et charger le modèle champion
mlflow.set_tracking_uri("sqlite:///mlflow.db")

try:
    model = mlflow.sklearn.load_model(
        "models:/churn-model@champion"
    )
except Exception as e:
    raise RuntimeError(
        f"Impossible de charger le modèle MLflow : {e}"
    ) from e


# 3. Définir les données attendues
class Client(BaseModel):
    age: int
    salaire: float
    abonnement: Literal["Basique", "Standard", "Premium"]
    anciennete: float
    consommation: float


# 4. Vérifier que l'API fonctionne
@app.get("/health")
def health():
    return {"status": "ok"}


# 5. Prédire le churn
@app.post("/predict")
def predict(client: Client):
    try:
        # Convertir les données en DataFrame
        ligne = pd.DataFrame([client.model_dump()])

        # Encoder la variable catégorielle
        ligne = pd.get_dummies(
            ligne,
            columns=["abonnement"],
            dtype=int
        )

        # Récupérer les colonnes attendues par le modèle
        colonnes = getattr(model, "feature_names_in_", None)

        if colonnes is None:
            raise ValueError(
                "Le modèle ne contient pas feature_names_in_."
            )

        # Aligner les colonnes et leur ordre
        ligne = ligne.reindex(
            columns=colonnes,
            fill_value=0
        )

        # Effectuer la prédiction
        resultat = int(model.predict(ligne)[0])

        return {
            "churn": resultat,
            "message": (
                "Le client va partir"
                if resultat == 1
                else "Le client reste"
            )
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la prédiction : {str(e)}"
        ) from e
