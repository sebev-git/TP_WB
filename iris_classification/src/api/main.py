import os
import joblib
import pandas as pd
import wandb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_NAME = "iris-classification"
ARTIFACT_TYPE = "sweep_model"

app = FastAPI(
    title="API de Classification d'Iris",
    description="API pour prédire l'espèce d'une fleur d'Iris à partir du meilleur artefact sweep_model.",
    version="2.1"
)

class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., alias="sepal length (cm)", example=5.1)
    sepal_width: float = Field(..., alias="sepal width (cm)", example=3.5)
    petal_length: float = Field(..., alias="petal length (cm)", example=1.4)
    petal_width: float = Field(..., alias="petal width (cm)", example=0.2)

class PredictionOut(BaseModel):
    predicted_class: str = Field(..., example="setosa")

def load_best_model():
    try:
        api = wandb.Api()
        runs = api.runs(PROJECT_NAME, order="-summary_metrics.test_accuracy")

        if not runs:
            raise RuntimeError("Aucun run trouvé dans le projet W&B.")

        for run in runs:
            for artifact in run.logged_artifacts():
                if artifact.type == ARTIFACT_TYPE:
                    print(f"Run sélectionné : {run.name} avec test_accuracy = {run.summary.get('test_accuracy')}")
                    print(f"Téléchargement de l'artefact : {artifact.name}")
                    artifact_dir = artifact.download()
                    joblib_files = [f for f in os.listdir(artifact_dir) if f.endswith(".joblib")]
                    if not joblib_files:
                        continue
                    model_path = os.path.join(artifact_dir, joblib_files[0])
                    print(f"Modèle chargé depuis : {model_path}")
                    return joblib.load(model_path)

        raise RuntimeError("Aucun artefact de type 'sweep_model' trouvé dans les meilleurs runs.")

    except Exception as e:
        print(f"[ERREUR] Chargement du modèle échoué : {e}")
        raise RuntimeError("Échec du chargement du modèle depuis W&B.") from e

model = load_best_model()

@app.get("/", tags=["Général"])
def read_root():
    return {"message": "Bienvenue sur l'API de classification d'Iris !"}

@app.post("/predict/", response_model=PredictionOut, tags=["Prédictions"])
def predict(features: IrisFeatures):
    try:
        data = pd.DataFrame([features.dict(by_alias=True)])
        prediction = model.predict(data)
        pred_class = str(prediction[0])
        return {"predicted_class": pred_class}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {e}")