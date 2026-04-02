import os
import wandb
import pandas as pd 
import numpy as np
import joblib
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# 1. Start a W&B Run
run = wandb.init(
    project="classification-car-accidents",
    name='My second run - With Artifacts',
    tags=["baseline", "Decision Tree"],
)

#  2. Capture a dictionary of hyperparameters
params = {"criterion": 'gini', "max_depth": 10}

wandb.config = params

X_train = pd.read_csv('data/preprocessed/X_train.csv')
X_test = pd.read_csv('data/preprocessed/X_test.csv')
y_train = pd.read_csv('data/preprocessed/y_train.csv')
y_test = pd.read_csv('data/preprocessed/y_test.csv')
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

le = LabelEncoder()
y_train = le.fit_transform(y_train)  # [2,3,4] → [0,1,2]
y_test = le.transform(y_test)        # même transformation

# Log datasets as artifact
data_artifact = wandb.Artifact(
    name="car-accident-dataset",
    type="dataset",
    description="Train/test splits used for second run"
)
data_artifact.add_file("data/preprocessed/X_train.csv")
data_artifact.add_file("data/preprocessed/X_test.csv")
data_artifact.add_file("data/preprocessed/y_train.csv")
data_artifact.add_file("data/preprocessed/y_test.csv")
wandb.log_artifact(data_artifact)

# 3. Train the model
dt_classifier = DecisionTreeClassifier(**params)
dt_classifier.fit(X_train, y_train)

# Save model locally and create W&B artifact
os.makedirs("models", exist_ok=True)
model_path = "models/dt_model.joblib"
joblib.dump(dt_classifier, model_path)

model_artifact = wandb.Artifact(
    name="decision_tree_model",
    type="model",
    description="Decision Tree trained with gini and max_depth=10",
)

model_artifact.add_file(model_path)
wandb.log_artifact(model_artifact)

# 4. Capture a dictionary of metrics
train_accuracy = dt_classifier.score(X_train, y_train)
test_accuracy = dt_classifier.score(X_test, y_test)
wandb.log({"train_accuracy": train_accuracy, "test_accuracy": test_accuracy})   

# 5. Track plots with sklearn.plot_classifier
y_pred = dt_classifier.predict(X_test)
y_probas = dt_classifier.predict_proba(X_test)
labels = ['non-prioritary accident', 'prioritary accident', 'critical accident']

wandb.sklearn.plot_classifier(
    dt_classifier,
    X_train,
    X_test,
    y_train,
    y_test,
    y_pred,
    y_probas,
    labels,
    model_name="Decision Tree",
    feature_names=X_train.columns,
)

time.sleep(60)

# Finish the run
wandb.finish()