import wandb
import os
import pandas as pd
import numpy as np
import yaml
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

run = wandb.init(project="iris-classification", job_type="data_loading")

artifact = run.use_artifact("iris_preprocessed_data:v0", type="dataset")
artifact_path = artifact.download()

X_train_path = os.path.join(artifact_path, "X_train.csv")
X_test_path = os.path.join(artifact_path, "X_test.csv")
y_train_path = os.path.join(artifact_path, "y_train.csv")
y_test_path = os.path.join(artifact_path, "y_test.csv")

X_train = pd.read_csv(X_train_path)
X_test = pd.read_csv(X_test_path)
y_train = np.ravel(pd.read_csv(y_train_path))
y_test = np.ravel(pd.read_csv(y_test_path))

run.finish()

def train():
    run = wandb.init()
    config = wandb.config

    model = DecisionTreeClassifier(
        criterion=config.criterion,
        max_depth=config.max_depth,
        random_state=config.random_state
    )
    model.fit(X_train, y_train)

    # Évaluation
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    wandb.log({"train_accuracy": train_acc, "test_accuracy": test_acc})

    # Sauvegarde du modèle
    os.makedirs("models", exist_ok=True)
    model_path = f"models/model_{run.id}.joblib"
    joblib.dump(model, model_path)

    # Journalisation du modèle en tant qu’artefact
    model_artifact = wandb.Artifact(
        name=f"dtree-model-{run.id}",
        type="sweep_model",
        description="Decision Tree model from sweep",
        metadata=dict(config)
    )
    model_artifact.add_file(model_path)
    run.log_artifact(model_artifact)

    run.finish()

# Initialize a temporary run to retrieve the config artifact
run = wandb.init(
    project="iris-classification",
    job_type="sweep_launcher",
    name="Load sweep config artifact"
)

artifact = run.use_artifact("iris_sweep_config:v0", type="config")
artifact_path = artifact.download()
config_file_path = os.path.join(artifact_path, "config_sweep.yaml")


# AJOUT : inspecter le contenu réel
print("Chemin du fichier :", config_file_path)
print("Fichier existe ?", os.path.exists(config_file_path))

# Load the YAML content of the configuration file
with open(config_file_path, "r") as f:
    sweep_config = yaml.safe_load(f)

run.finish()

sweep_id = wandb.sweep(sweep_config, project="iris-classification")
wandb.agent(sweep_id, function=train, count=5)