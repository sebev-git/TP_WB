import wandb
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib
import time

run = wandb.init(
    project="classification-car-accidents",
    name='My first run',
    tags=["baseline", "random-forest"],
)

# Définir les hyperparamètres
params = {"n_estimators": 100, "criterion": "gini", "max_depth": 10}

# Enregistrer les hyperparamètres dans W&B
wandb.config = params

X_train = pd.read_csv('data/preprocessed/X_train.csv')
X_test = pd.read_csv('data/preprocessed/X_test.csv')
y_train = pd.read_csv('data/preprocessed/y_train.csv')
y_test = pd.read_csv('data/preprocessed/y_test.csv')
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

rf_classifier = RandomForestClassifier(**params)

rf_classifier.fit(X_train, y_train)

train_accuracy = rf_classifier.score(X_train, y_train)
test_accuracy = rf_classifier.score(X_test, y_test)
wandb.log({"train_accuracy": train_accuracy, "test_accuracy": test_accuracy})

# Sauvegarder le modèle localement
model_filename = './src/models/trained_model.joblib'
joblib.dump(rf_classifier, model_filename)
print("Model trained and saved successfully.")

time.sleep(60)
wandb.finish()