
# ===== 1. Importer les bibliothèques =====
import mlflow
import mlflow.sklearn

# ===== 2. Se connecter à la base du Registry =====
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# ===== 3. Charger le modèle Champion depuis le Registry =====
model = mlflow.sklearn.load_model(
    "models:/churn-model@champion"
)

# ===== 4. Exporter le modèle dans un dossier local =====
mlflow.sklearn.save_model(
    model,
    path="model",
    serialization_format="cloudpickle"
)

print("Champion exporté dans le dossier model/")
